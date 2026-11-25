"""Five observed surface clicks -> positive base-Z hover, no camera/CAN access."""
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from eyehand.click_hover import HoverPlanner,load_config,hover_target,in_workspace
from scripts.click_hover_eth import project_base_pixel
from eyehand.table_reference import table_hover_target,effective_hover_m

p=HoverPlanner(load_config())
assert p.cfg['hover_m']>0 and p.cfg['height_reference']=='table'
table_z=0.  # Synthetic reference for offline reachability checks.
assert p.cfg['workspace_min_m'][2]==.03
points=[([847,547],[.11385989178141619,.09981490394002324,.545000025886111]),
        ([567,558],[-.054258538202759,.1077747252096695,.5520000262185931]),
        ([911,538],[.15436278673921697,.09580652848583616,.5530000262660906]),
        ([885,574],[.13532953345582105,.11493379716354234,.5400000256486237]),
        ([505,432],[-.09463602196152607,.032247063859478814,.5690000270260498])]
for pixel,point in points:
    surface,target=table_hover_target(p.T_base_camera,point,effective_hover_m(p.cfg),table_z_m=table_z)
    np.testing.assert_array_equal(target[:2],surface[:2])
    np.testing.assert_allclose(target[2]-table_z,effective_hover_m(p.cfg),atol=1e-12)
    assert in_workspace(target,p.cfg)
    np.testing.assert_allclose(project_base_pixel(p,surface),pixel,atol=1e-8)
    plan=p.plan(np.zeros(6),target)
    assert np.linalg.norm(p.tcp(plan.goal)-target)<.0002
    qs=np.array([plan.sample(t)[0] for t in np.linspace(0,plan.duration,301)])
    assert np.all(qs>=p.raw_lower-1e-12) and np.all(qs<=p.raw_upper+1e-12)
    assert all(in_workspace(p.tcp(q),p.cfg) for q in qs)
    assert p.plan_home(plan.goal).goal.tolist()==[0.]*6
print('Five recorded clicks: unchanged base XY, fixed table-relative height, projected surface/hover separation, SDK IK, trajectory bounds and home passed; no hardware')
