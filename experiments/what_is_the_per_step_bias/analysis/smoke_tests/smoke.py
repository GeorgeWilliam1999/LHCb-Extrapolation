import os, sys, time, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common import *
import stage_equations as se
setup_threads()
t=time.time()
e = cached_single_step_errors("25dfa17ef168", "test")
print("e", e.shape, time.time()-t, lag_one_scores(e))
net = network_of("25dfa17ef168")
fin = chain(plain_step(net), split("test").start_state)
print(endpoint_scores(fin[:, -1])[0], time.time()-t)
ex = chain(lambda s, z: se.exact_step(s, z)[0], split("test").start_state)
print("exact", endpoint_scores(ex[:, -1])[0]["endpoint_radial_median_um"], time.time()-t)
