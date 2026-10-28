import os, sys, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common import *
import stage_equations as se
setup_threads()
net = network_of("25dfa17ef168")
test = split("test")
s0 = test.start_state[:200]
ref_end = test.reference_end_state[:200]
def score(fin):
    d=(fin[:, :2]-ref_end[:, :2])*1e3; return np.median(np.hypot(d[:,0],d[:,1]))
print("plain", score(chain(plain_step(net), s0)[:, -1]))
for sw, nw in ((1,0),(2,0),(0,1),(0,3)):
    def step(s, z):
        st = stages_and_end(net, s, z)[0]
        return se.polish(s, z, st, sweeps=sw, newton_steps=nw)[0]
    print("sweeps", sw, "newton", nw, score(chain(step, s0)[:, -1]))
print("exact", score(chain(lambda s,z: se.exact_step(s,z)[0], s0)[:, -1]))
