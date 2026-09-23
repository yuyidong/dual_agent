import importlib.util
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT/'scripts/experiments/pv_station_mechanistic_consistency/station_consistency.py'
spec = importlib.util.spec_from_file_location('station_experiment', SCRIPT)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_perturbation_is_symmetric_and_station_local():
    base = torch.tensor([[[[20.,30.],[3.,40.],[80.,90.]]]])
    capacity = torch.tensor([100.,100.])
    mask = m.common_perturbation_mask(base,capacity,torch.ones_like(base),.05,capacity)
    assert torch.equal(mask[...,0],mask[...,1])
    assert mask[0,0,1].sum() == 0
    saved = base.clone()
    plus = m.perturb(base,capacity,mask,1,.05,1)
    minus = m.perturb(base,capacity,mask,1,.05,-1)
    assert torch.equal(base,saved)
    assert torch.equal(plus[...,0],base[...,0])
    assert torch.equal(minus[...,0],base[...,0])
    assert torch.equal(plus-base,base-minus)
    assert torch.all(plus <= capacity) and torch.all(minus >= 0)


def test_existing_out_of_bounds_prediction_is_not_changed():
    base = torch.tensor([[[[120.,30.],[50.,50.]]]])
    capacity = torch.tensor([100.,100.])
    mask = m.common_perturbation_mask(base,capacity,torch.ones_like(base),.05,capacity)
    assert mask[0,0,0].sum() == 0
    changed = m.perturb(base,capacity,mask,0,.05,-1)
    assert changed[0,0,0,0] == 120
    assert changed[0,0,1,0] == 45


def test_night_is_excluded_even_with_headroom():
    base = torch.ones(1,2,3,4)*50
    capacity = torch.ones(4)*100
    daylight = torch.ones(1,1,3,4)
    daylight[:,:,1,:] = 0
    mask = m.common_perturbation_mask(base,capacity,daylight,.05,capacity)
    assert mask[:,:,1,:].sum() == 0
    assert torch.all(mask[:,:,0,:] == 1)


def test_numerical_noise_does_not_create_station_ranking():
    import numpy as np
    tied = np.array([[146.63470857961585,146.63470857961582,146.6347085796156,146.63470857961548], [486.766025,486.772785,486.770393,486.774621]])
    assert m.ranking_agreement(tied) is None
    assert np.all(m.station_ranks(tied[0]) == 2.5)
    assert m.ranking_agreement([[4,3,2,1],[8,6,4,2]]) == 1.0
