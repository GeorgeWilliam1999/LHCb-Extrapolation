"""Tool (K=12 2x64, no fallback) for most tracks; plain K Nystrom steps for tracks whose straight line
is above a |y| cut at z = 5000 mm (a rule on the start state, no ARKN). VALIDATION tracks."""
import sys, os, numpy as np
sys.path.insert(0,"python")
import an_tools as an, fallback_validation as fv
from tail_diagnosis import radial
s,ref,p=fv.validation_tracks()
W="tool:"+os.path.join(an.WEIGHTS,"linear_nystrom_12_2x64.json")+":fallback=1e30"
e0,_,c0=an.run(W,s,an.Z0,an.Z1); r0=radial(e0,ref)
y5=np.abs(s[:,1]+s[:,3]*(5000-an.Z0))
print("tool alone: median %.2f 99.9%% %.0f max %.0f reads %.1f"%(np.median(r0),np.percentile(r0,99.9),r0.max(),c0[:,0].mean()))
for K in (24,32,48,64):
    e1,_,c1=an.run("nystrom:%d"%K,s,an.Z0,an.Z1); r1=radial(e1,ref)
    for cut in (500,600,700):
        m=y5>cut; r=np.where(m,r1,r0); reads=np.where(m,c1[:,0],c0[:,0]).mean()
        print("|y@5m|>%d (%.1f%% of tracks) -> plain Nystrom %d: median %.2f 95%% %.1f 99.9%% %.0f max %.0f >1mm %d reads %.1f"%(cut,100*m.mean(),K,np.median(r),np.percentile(r,95),np.percentile(r,99.9),r.max(),(r>1000).sum(),reads))
