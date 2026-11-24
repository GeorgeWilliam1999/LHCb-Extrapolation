"""Print the core to time on: the core of this node whose cache group is the idlest.

The cores of a node share their largest cache in groups (four cores on the
EPYC 7702P of stbc-i3). A busy neighbour in the group takes cache from the
timed program, so the group is chosen, not the core alone: the group with the
highest idle time over 5 seconds, and in it the idlest core. Only groups whose
cores are all open to this user are taken.

    quietest_core.py        prints one core number
"""
from __future__ import annotations

import glob
import os
import time


def idle_and_total():
    cores = {}
    for line in open("/proc/stat"):
        parts = line.split()
        if parts[0].startswith("cpu") and parts[0] != "cpu":
            values = [int(v) for v in parts[1:]]
            cores[int(parts[0][3:])] = (values[3] + values[4], sum(values))
    return cores


def groups():
    found = {}
    for path in glob.glob("/sys/devices/system/cpu/cpu[0-9]*/cache/index3/shared_cpu_list"):
        text = open(path).read().strip()
        members = []
        for part in text.split(","):
            low, _, high = part.partition("-")
            members += list(range(int(low), int(high or low) + 1))
        found[text] = members
    return list(found.values())


if __name__ == "__main__":
    before = idle_and_total()
    time.sleep(5)
    after = idle_and_total()
    idle = {c: (after[c][0] - before[c][0]) / max(1, after[c][1] - before[c][1]) for c in after}
    allowed = os.sched_getaffinity(0)      # a login node gives a user only some of its cores
    usable = [members for members in groups() if set(members) <= allowed]
    best = max(usable, key=lambda members: sum(idle.get(c, 0.0) for c in members) / len(members))
    print(max(best, key=lambda c: idle.get(c, 0.0)))
