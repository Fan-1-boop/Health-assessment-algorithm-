import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
import pandas as pd
from v14_context import TargetRidgeWithCommands
from extract_esa_commands import select_commands,pulses_asof


def test_command_pulse_never_looks_forward():
    ticks=pd.date_range('2022-01-01',periods=5,freq='10min',tz='UTC')
    events=[ticks[1]+pd.Timedelta(minutes=1)]
    vals=pulses_asof(ticks,events,window='10min')
    assert vals.tolist()==[0,0,1,0,0]


def test_command_priority_and_size_selection_is_deterministic():
    catalog=pd.DataFrame({'Telecommand':['x','y','z','a'],
                          'Priority':[3,3,2,4]})
    sizes={'ESA-Mission1/telecommands/x.zip':100,
           'ESA-Mission1/telecommands/y.zip':200,
           'ESA-Mission1/telecommands/z.zip':300,
           'ESA-Mission1/telecommands/a.zip':300}
    first=select_commands(catalog,sizes,count=2)
    second=select_commands(catalog,sizes,count=2)
    assert first==second
    assert [x['Telecommand'] for x in first]==['a','y']


def test_context_aware_ridge_returns_finite_anomaly_scores():
    rng=np.random.default_rng(42)
    n=90
    w=np.zeros((n,16,3))
    w[:,:,0]=rng.normal(size=(n,16))
    w[:,:,1:]=(rng.random((n,16,2))>.92)
    model=TargetRidgeWithCommands().fit(w[:60])
    scores=model.score(w[60:])
    assert scores.shape==(30,)
    assert np.isfinite(scores).all()
    assert (scores>=0).all()
