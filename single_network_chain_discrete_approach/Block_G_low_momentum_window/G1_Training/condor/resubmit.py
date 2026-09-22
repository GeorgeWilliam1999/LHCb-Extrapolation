#!/usr/bin/env python
"""Send the Block G runs that have not finished back to the farm to continue.

Block F's `F1_Training/resubmit.py` with the momentum window added to the queue
match and to the run folder, and with the two safety rules below. A run is
finished when its `progress.json` says "done" AND it has used the restart budget
of THIS job line. `record.json` alone is not enough: a run waiting to be extended
still carries the record of its previous stopping point, and a line with a larger
--outer-cap is a request to train it further.

The queue match is (wrapper, N, q, weighting, window). The executable has to be
matched as well as the arguments: Block E and Block F run the same (N, q)
through their own wrappers, so matching on --N/--q alone would read their jobs
as this block's. The window is in the match because two windows are two
different runs of the same (N, q, weighting) in two different folders.

TWO RULES THAT BLOCK F DID NOT HAVE. Both exist because on 2026-09-21 at 13:35
Block F's keeper submitted a second copy of its N = 256, q = 16 job and the two
processes have been writing one checkpoint ever since:

  1. an unreadable queue means UNKNOWN, not empty. `condor_q` failing - a
     non-zero exit, an exception, or an error on stderr - used to return an
     empty set, which reads as "nothing of ours is running" and resubmits
     everything. Here it returns None and NOTHING is submitted. A queue we
     cannot read is not a queue we can act on; the next pass half an hour later
     will read it.
  2. a run whose `progress.json` was written in the last LIVE_SECONDS is
     treated as alive even if the queue does not show it, and is not
     resubmitted. The trainer writes progress.json after every restart (78 s at
     N = 64, 226 s at N = 256), so a file younger than ten minutes means a
     process is in the middle of that run. This is the belt to rule 1's braces:
     it catches a job that is running but missing from the queue listing for any
     reason at all.

    python resubmit.py             # list only
    python resubmit.py --submit    # one pass
    python resubmit.py --self-test # the two rules, on a fake condor_q
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))       # .../G1_Training/condor
ROOT = os.path.dirname(HERE)                            # .../G1_Training
sys.path.insert(0, HERE)
from make_jobs import SUB, job_lines                    # noqa: E402

KEY = re.compile(r"--N\s+(\d+)\s+--q\s+(\d+)")
WEIGHTING = re.compile(r"--weighting\s+(\w+)")
P_LO = re.compile(r"--p-lo\s+([0-9.eE+-]+)")
P_HI = re.compile(r"--p-hi\s+([0-9.eE+-]+)")
OUT = re.compile(r"--out\s+(\S+)")
CAP = re.compile(r"--outer-cap\s+(\d+)")
WRAPPER = os.path.join(HERE, "wrapper.sh")
LIVE_SECONDS = 600                  # a progress.json younger than this means a live process
QUEUE_ERROR = re.compile(r"error|failed|unable|cannot|can't|denied|refused", re.I)
DEFAULT_WINDOW = (10.0, 50.0)       # train_windowed.py's own defaults


def key_of(line, default_weighting="full"):
    """(N, q, weighting, p_lo, p_hi) - what identifies one run, in a job line or the queue."""
    m = KEY.search(line)
    if not m:
        return None
    w = WEIGHTING.search(line)
    lo, hi = P_LO.search(line), P_HI.search(line)
    return (int(m.group(1)), int(m.group(2)), w.group(1) if w else default_weighting,
            float(lo.group(1)) if lo else DEFAULT_WINDOW[0],
            float(hi.group(1)) if hi else DEFAULT_WINDOW[1])


def run_dir(line):
    """The run folder a job line writes into: <out>/<weighting>/N<NNN>_q<qq>/."""
    k = key_of(line)
    if k is None:
        return None
    N, q, w = k[0], k[1], k[2]
    o = OUT.search(line)
    out = o.group(1) if o else "results"
    if not os.path.isabs(out):
        out = os.path.join(ROOT, out)
    return os.path.join(out, w, "N%03d_q%02d" % (N, q))


def queued(run=subprocess.run):
    """The keys of THIS block's jobs in the queue, or None if the queue could not be read.

    None is not an empty set. Every caller must treat it as "unknown" and do
    nothing; see rule 1 in the module docstring.
    """
    try:
        r = run(["condor_q", "-af", "Cmd", "Args"], capture_output=True, text=True)
    except Exception as e:                                  # condor_q missing, killed, ...
        print("condor_q could not be run (%s: %s)" % (type(e).__name__, e))
        return None
    if getattr(r, "returncode", 1) != 0:
        print("condor_q exited %s" % getattr(r, "returncode", "?"))
        return None
    err = (getattr(r, "stderr", "") or "").strip()
    if err and QUEUE_ERROR.search(err):
        print("condor_q reported an error: %s" % err.splitlines()[0])
        return None
    out = getattr(r, "stdout", "") or ""
    return {k for k in (key_of(line) for line in out.splitlines() if WRAPPER in line) if k}


def live(line, now=None, live_seconds=LIVE_SECONDS):
    """Seconds since this run's progress.json was written, or None if there is none."""
    d = run_dir(line)
    prog = os.path.join(d, "progress.json") if d else None
    if not (prog and os.path.exists(prog)):
        return None
    return (time.time() if now is None else now) - os.path.getmtime(prog)


def finished(line):
    """Has this run reached the restart budget of THIS job line?"""
    d = run_dir(line)
    prog = os.path.join(d, "progress.json") if d else None
    if not (prog and os.path.exists(prog)):
        return False
    cap = CAP.search(line)
    cap = int(cap.group(1)) if cap else 400
    pr = json.load(open(prog))
    return pr.get("phase") == "done" and pr.get("restart", 0) >= cap


def plan(lines, inq, now=None, live_seconds=LIVE_SECONDS):
    """(the lines to resubmit, the lines held back with the reason).

    `inq` is what `queued()` returned: a set of keys, or None for "unknown".
    """
    if inq is None:
        return [], [("(every line)", "the queue could not be read; nothing is submitted")]
    todo, held = [], []
    for line in [x for x in lines if x.strip()]:
        k = key_of(line)
        if k is None:
            continue
        if finished(line):
            held.append((line, "finished: at its restart cap"))
            continue
        if k in inq:
            held.append((line, "already in the queue"))
            continue
        age = live(line, now=now, live_seconds=live_seconds)
        if age is not None and age < live_seconds:
            held.append((line, "a process wrote its progress %.0f s ago" % age))
            continue
        todo.append(line)
    return todo, held


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--memory-mb", type=int, default=8192)
    ap.add_argument("--category", default="medium")
    ap.add_argument("--jobs-from", default=os.path.join(HERE, "jobs_active.txt"),
                    help="the job lines to keep alive (default: condor/jobs_active.txt if it "
                         "exists, else the standard three runs)")
    ap.add_argument("--self-test", action="store_true",
                    help="check the two safety rules on a fake condor_q and exit")
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()
    lines = (open(a.jobs_from).read().split("\n") if os.path.exists(a.jobs_from) else job_lines())
    todo, held = plan(lines, queued())
    for line, why in held:
        print("  held back: %s   [%s]" % (why, line.strip()))
    print("%d runs unfinished and not queued" % len(todo))
    if not todo:
        return []
    rnd = 1
    while os.path.exists(os.path.join(HERE, "jobs_round%d.txt" % rnd)):
        rnd += 1
    jobs, sub = "condor/jobs_round%d.txt" % rnd, "condor/jobs_round%d.sub" % rnd
    with open(os.path.join(ROOT, jobs), "w") as f:
        f.write("\n".join(todo) + "\n")
    with open(os.path.join(ROOT, sub), "w") as f:
        f.write(SUB.format(wrapper=WRAPPER, memory=a.memory_mb, category=a.category, jobs=jobs))
    for line in todo:
        print("   " + line)
    if a.submit:
        subprocess.run(["condor_submit", sub], cwd=ROOT, check=True)
    else:
        print("not submitted: %s" % sub)
    return todo


# ------------------------------------------------------------- the self-test --
def self_test():
    """The two rules, on a fake `condor_q` and a fake run folder. Writes nothing real."""
    import tempfile
    ok = []

    def check(name, cond):
        ok.append(bool(cond))
        print("  %-58s %s" % (name, "pass" if cond else "FAIL"))

    class R:
        def __init__(self, rc=0, out="", err=""):
            self.returncode, self.stdout, self.stderr = rc, out, err

    with tempfile.TemporaryDirectory() as tmp:
        line = ("--N 64 --q 2 --weighting full --p-lo 3 --p-hi 8 --out %s "
                "--round-restarts 25 --round-cap 40 --outer-cap 1000" % tmp)
        other = line.replace("--p-lo 3 --p-hi 8", "--p-lo 10 --p-hi 50")
        good = "%s %s\n" % (WRAPPER, line)

        # -- rule 1: a queue that cannot be read means unknown, and nothing is submitted
        check("condor_q exits non-zero -> unknown",
              queued(run=lambda *a, **k: R(rc=1, err="condor_q: cannot connect")) is None)
        check("condor_q raises -> unknown",
              queued(run=lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError("condor_q")))
              is None)
        check("condor_q exits 0 with an error on stderr -> unknown",
              queued(run=lambda *a, **k: R(rc=0, out="", err="Error: SCHEDD unreachable")) is None)
        check("unknown queue -> nothing is submitted",
              plan([line], None)[0] == [])
        check("empty queue, read cleanly -> the line IS submitted",
              queued(run=lambda *a, **k: R(rc=0, out="")) == set()
              and plan([line], set())[0] == [line])
        check("a warning on stderr that is not an error -> still read",
              queued(run=lambda *a, **k: R(rc=0, out=good, err="Warning: 1 slot unclaimed"))
              == {key_of(line)})

        # -- the queue match: window, weighting and wrapper all have to agree
        inq = queued(run=lambda *a, **k: R(rc=0, out=good))
        check("the line's own job in the queue -> held back", plan([line], inq)[0] == [])
        check("the same N, q at another window -> NOT a match", plan([other], inq)[0] == [other])
        foreign = "/some/other/block/condor/wrapper.sh %s\n" % line
        check("the same arguments under another wrapper -> not ours",
              queued(run=lambda *a, **k: R(rc=0, out=foreign)) == set())

        # -- rule 2: a run whose progress.json is warm is alive
        d = os.path.join(tmp, "full", "N064_q02")
        os.makedirs(d)
        prog = os.path.join(d, "progress.json")
        json.dump(dict(phase="train", restart=7), open(prog, "w"))
        check("progress.json written just now -> held back", plan([line], set())[0] == [])
        os.utime(prog, (time.time() - 3600, time.time() - 3600))
        check("progress.json an hour old -> submitted", plan([line], set())[0] == [line])
        json.dump(dict(phase="done", restart=1000), open(prog, "w"))
        os.utime(prog, (time.time() - 3600, time.time() - 3600))
        check("finished at its restart cap -> held back", plan([line], set())[0] == [])

    print("%d of %d checks pass" % (sum(ok), len(ok)))
    if not all(ok):
        raise SystemExit(1)
    return ok


if __name__ == "__main__":
    main()
