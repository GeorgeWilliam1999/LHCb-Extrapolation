import os, sys
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)) or "."
V1 = os.path.join(HERE, "..", "S2_One_step_network")
sys.path.insert(0, V1); sys.path.insert(0, os.path.join(HERE, "..", "S0_Baseline_data_exploration"))
sys.path.insert(0, os.path.join(HERE, "..", "S1_Simple_first_pass"))
from irk import tableau
from model import LHCbRates, OneStepNetwork, reconstruction_residuals
from reference_card import FIELD
torch.set_default_dtype(torch.float64)
rates = LHCbRates()
d = np.load("results/frozen_leg_data.npz")
Q = int(d["q"]); DZ = float(d["z1"]) - float(d["z0"])
c, A_np, b_np = tableau(Q); A, b = torch.tensor(A_np), torch.tensor(b_np)
S = torch.tensor(d["train_S"]); ref = d["train_ref"]; P = d["train_P"]
lo = FIELD.min; hi = [FIELD.min[i] + (FIELD.N[i]-1)/FIELD.invD[i] for i in range(3)]
out = (((ref[:,:,0] < lo[0]) | (ref[:,:,0] > hi[0]) | (ref[:,:,1] < lo[1]) | (ref[:,:,1] > hi[1])).any(axis=1))
m = OneStepNetwork(Q, d["in_scale"], d["out_scale"])
m.load_state_dict(torch.load("results/one_step_physics_seed0.pt", weights_only=True))
with torch.no_grad():
    r = reconstruction_residuals(m, rates, S, DZ, torch.tensor(d["znodes"]), A, b).numpy()
ps = (r**2).mean(axis=(1,2))
top = np.argsort(ps)[::-1][:20]
print("out-of-map states: %d | of the 20 worst-loss states, %d are out-of-map"
      % (out.sum(), out[top].sum()))
print("loss share of out-of-map states: %.0f%%" % (100*ps[out].sum()/ps.sum()))
print("their |p|: %s GeV" % np.round(np.sort(P[out])[:8], 2))
print("median |p| of all: %.2f GeV" % np.median(P))
keep = ~out
print("loss if they were excluded: %.3e  (vs %.3e)  -> %.0fx lower"
      % (ps[keep].mean(), ps.mean(), ps.mean()/ps[keep].mean()))
