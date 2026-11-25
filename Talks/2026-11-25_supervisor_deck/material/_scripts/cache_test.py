"""Copy the test split of both track sets out of the 1 GB files once (read only)."""
import numpy as np, os
from style import WORK, STORE
for key in ("5558be31abeb", "afc61a558211"):
    z = np.load(f"{STORE}/tracks/{key}/tracks.npz")
    out = {k.split(".", 1)[1]: z[k] for k in z.files if k.startswith("test.")}
    out["planes_mm"] = z["planes_mm"]
    np.savez(os.path.join(WORK, f"test_{key}.npz"), **out)
    print(key, {k: v.shape for k, v in out.items()})
