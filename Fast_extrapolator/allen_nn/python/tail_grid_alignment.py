import sys, numpy as np
sys.path.insert(0,"python")
import an_tools as an, fallback_validation as fv
from tail_diagnosis import Field, planes_spec, radial
s,ref,p=fv.validation_tracks(); f=Field()
# y asymmetry of the map at x=0
for z in (4000,4400,4800,5500):
    ys=np.array([-1300,-1000,-800,0,800,1000,1300.]); B=f(np.zeros(7),ys,np.full(7,z))
    print("z=%d By(y)="%z, dict(zip(ys.astype(int), np.round(B[:,1],3))), " Bz:", np.round(B[:,2],3))
# grid-aligned planes: interior planes rounded to the 100 mm grid nodes (lowest corner z=-500)
def score(name,pl):
    e,_,c=an.run(planes_spec(pl),s,an.Z0,an.Z1); r=radial(e,ref)
    print("%-44s K=%d reads %.0f median %.2f 95%% %.1f 99.9%% %.0f max %.0f >1mm %d"%(name,len(pl)-1,c[:,0].mean(),np.median(r),np.percentile(r,95),np.percentile(r,99.9),r.max(),(r>1000).sum()))
for K in (12,16,20,24):
    u=an.Z0+np.arange(K+1)*(an.Z1-an.Z0)/K
    a=u.copy(); a[1:-1]=np.round((a[1:-1]+500)/100)*100-500
    score("uniform",u); score("uniform, planes on grid nodes",a)
    h=u.copy(); h[1:-1]=np.round((h[1:-1]+500-50)/100)*100-500+50
    score("uniform, planes mid-cell",h)
