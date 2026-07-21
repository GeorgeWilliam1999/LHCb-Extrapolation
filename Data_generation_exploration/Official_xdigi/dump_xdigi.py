#!/usr/bin/env python
"""Dump the MC truth of official TestFileDB .xdigi files to plain CSVs.

Adapted from First_Pass/dump_event.py (the .sim dump) for the centrally
produced `expected_2024_minbias_xdigi` sample. XDIGI = extended digi: the
packed MC truth (pSim/...) travels with the digitised banks, so the same
GaudiConf.SimConf(EnableUnpack=True) Data-On-Demand recipe unpacks it.

Differences from the .sim dump:
  - input is a LIST of files (one concatenated event stream; the `evt` column
    is the 0-based index in the stream, unique across files);
  - default inputs are the local CVMFS testfiledb-mirror copies (no EOS/auth);
  - stops cleanly at end of stream.

Run (same environment recipe as the .sim dump):
    lb-run Gauss/v61r0p2 python dump_xdigi.py <outdir> <n_events> [file1 file2 ...]
"""
import csv
import os
import sys

MIRROR = "/cvmfs/lhcbdev.cern.ch/testfiledb-mirror/lhcb/swtest/expected_2024_minbias_xdigi"
DEFAULT_FILES = [
    os.path.join(MIRROR, f)
    for f in [
        "00212966_00000013_1.xdigi",
        "00212966_00000018_1.xdigi",
        "00212966_00000026_1.xdigi",
        "00212966_00000031_1.xdigi",
        "00212966_00000086_1.xdigi",
    ]
]

outdir = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else "truth_official")
nevents = int(sys.argv[2]) if len(sys.argv) > 2 else 10
infiles = [os.path.abspath(f) for f in sys.argv[3:]] or DEFAULT_FILES
os.makedirs(outdir, exist_ok=True)
for f in infiles:
    assert os.path.exists(f), "input not found: %s" % f

from Configurables import (  # noqa: E402
    ApplicationMgr,
    EventDataSvc,
    EventSelector,
    LHCbApp,
    PersistencySvc,
)
from GaudiConf.SimConf import SimConf  # noqa: E402

LHCbApp(DataType="2024", Simulation=True, EnableHive=False)
SimConf(EnableUnpack=True)  # Data-On-Demand unpacking of pSim -> MC locations

# --- what GaudiConf.IOHelper('ROOT') used to set up ------------------------
ApplicationMgr().ExtSvc += [
    "Gaudi::MultiFileCatalog/FileCatalog",
    "Gaudi::IODataManager/IODataManager",
    "Gaudi::RootCnvSvc/RootCnvSvc",
    "DataOnDemandSvc",
]
PersistencySvc("EventPersistencySvc").CnvServices += ["Gaudi::RootCnvSvc/RootCnvSvc"]
EventDataSvc(ForceLeaves=True, RootCLID=1, EnableFaultHandler=True)
EventSelector().Input = [
    "DATAFILE='PFN:%s' SVC='Gaudi::RootEvtSelector' OPT='READ'" % f for f in infiles
]
# ----------------------------------------------------------------------------

import GaudiPython  # noqa: E402

appMgr = GaudiPython.AppMgr()
evt = appMgr.evtsvc()

summary = []
fp = open(os.path.join(outdir, "particles.csv"), "w", newline="")
fv = open(os.path.join(outdir, "vertices.csv"), "w", newline="")
fh = open(os.path.join(outdir, "hits.csv"), "w", newline="")
fc = open(os.path.join(outdir, "collisions.csv"), "w", newline="")
wp, wv, wh, wc = (csv.writer(f) for f in (fp, fv, fh, fc))
wp.writerow(
    [
        "evt", "key", "pid", "charge3", "px", "py", "pz", "E", "m",
        "ox", "oy", "oz", "ot", "origin_vtx_type",
        "ex", "ey", "ez",
        "mother_key", "pv_key", "n_end_vertices",
    ]
)
wv.writerow(["evt", "key", "x", "y", "z", "t", "type", "is_primary", "mother_key", "n_products"])
wh.writerow(
    [
        "evt", "det", "mc_key", "pid",
        "entry_x", "entry_y", "entry_z",
        "exit_x", "exit_y", "exit_z",
        "energy_MeV", "time_ns", "path_mm", "sensDetID",
    ]
)
wc.writerow(["evt", "index", "process_type", "is_signal"])

for ievt in range(nevents):
    appMgr.run(1)
    parts = evt["MC/Particles"]
    if not parts:
        print("No MC/Particles at stream index %d - end of stream." % ievt)
        break

    summary.append(("evt %d MC/Particles" % ievt, parts.size()))
    for p in parts:
        ov = p.originVertex()
        pos = ov.position() if ov else None
        endvs = p.endVertices()
        if endvs.size() > 0:
            ep = endvs[0].target().position()
            ex, ey, ez = ep.x(), ep.y(), ep.z()
        else:
            ex = ey = ez = ""
        mom = p.mother()
        pv = p.primaryVertex()
        m4 = p.momentum()
        wp.writerow(
            [
                ievt, p.key(), p.particleID().pid(), p.particleID().threeCharge(),
                m4.px(), m4.py(), m4.pz(), m4.e(), m4.M(),
                pos.x() if pos else "", pos.y() if pos else "",
                pos.z() if pos else "", ov.time() if ov else "",
                int(ov.type()) if ov else "",
                ex, ey, ez,
                mom.key() if mom else "", pv.key() if pv else "",
                endvs.size(),
            ]
        )

    verts = evt["MC/Vertices"]
    summary.append(("evt %d MC/Vertices" % ievt, verts.size() if verts else 0))
    if verts:
        for v in verts:
            pos = v.position()
            mo = v.mother()
            wv.writerow(
                [
                    ievt, v.key(), pos.x(), pos.y(), pos.z(), v.time(),
                    int(v.type()), int(v.isPrimary()),
                    mo.key() if mo else "", v.products().size(),
                ]
            )

    for det in ["VP", "UT", "FT", "Muon"]:
        hits = evt["MC/%s/Hits" % det]
        summary.append(("evt %d MC/%s/Hits" % (ievt, det), hits.size() if hits else 0))
        if not hits:
            continue
        for h in hits:
            en, ex_ = h.entry(), h.exit()
            mcp = h.mcParticle()
            wh.writerow(
                [
                    ievt, det,
                    mcp.key() if mcp else "",
                    mcp.particleID().pid() if mcp else "",
                    en.x(), en.y(), en.z(),
                    ex_.x(), ex_.y(), ex_.z(),
                    h.energy(), h.time(), h.pathLength(), h.sensDetID(),
                ]
            )

    colls = evt["Gen/Collisions"]
    summary.append(("evt %d Gen/Collisions" % ievt, colls.size() if colls else 0))
    if colls:
        for i, c in enumerate(colls):
            wc.writerow([ievt, i, c.processType(), int(c.isSignal())])

for f in (fp, fv, fh, fc):
    f.close()

with open(os.path.join(outdir, "tes_summary.txt"), "w") as f:
    f.write("Inputs:\n%s\n\n" % "\n".join(infiles))
    for loc, n in summary:
        f.write("%-28s %s\n" % (loc, n))

print("\n".join("%-28s %s" % (loc, n) for loc, n in summary[-8:]))
print("CSV files written to", outdir)
