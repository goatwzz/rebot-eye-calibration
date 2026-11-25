"""Historical-log regressions retain their original raw-angle calibration model."""
from pathlib import Path
from wrist.click_hover import load_config as load_active_config
ROOT=Path(__file__).resolve().parents[1]
def load_legacy_config(path=ROOT/'config/click_hover.json'):
    cfg=load_active_config(path)
    cfg['result']=str(ROOT/'tests/fixtures/manual50.npz')
    return cfg
