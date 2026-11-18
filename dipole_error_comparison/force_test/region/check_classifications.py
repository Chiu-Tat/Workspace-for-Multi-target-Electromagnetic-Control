"""Independent pointwise spot checks of whole-box verdicts (not their proof)."""
from .region_model import *
from .run_regions import reserve,P
from scipy.optimize import linprog
import json
O=Path(__file__).resolve().parent;A=np.load(O/'boxes.npy');rng=np.random.default_rng(1846)
scale=np.array([.02,.02,.1,.1]);checks=0
for mode in range(3):
 for verdict in [-1,1]:
  ids=np.where(A[:,9+mode]==verdict)[0]
  if len(ids)>30:ids=rng.choice(ids,30,replace=False)
  for i in ids:
   q=A[i]
   for _ in range(3):
    v=rng.uniform(q[[0,2,4]],q[[1,3,5]])
    g,c1,c5=maps(np.repeat(v,2).tolist(),P)
    gg=np.array([[x.center for x in row]for row in g])/scale[:,None]
    cc=np.array([x.center for x in (c1 if mode==0 else c5)])/scale
    eps=reserve(np.hypot(v[0]-.02,v[1])*1e3) if mode==2 else 0
    feasible=True
    for h in task(eps)/scale:
      r=linprog(np.zeros(10),A_eq=gg,b_eq=h-cc,bounds=[(-15,15)]*10,method='highs')
      if r.status==2:feasible=False;break
      assert r.status==0,r.message
    assert feasible==(verdict==1),(mode,verdict,i,v)
    checks+=1
j=json.loads((O/'validation.json').read_text());j['independent_pointwise_verdict_checks']=checks;(O/'validation.json').write_text(json.dumps(j,indent=2));print(checks,'pointwise verdict checks passed')
