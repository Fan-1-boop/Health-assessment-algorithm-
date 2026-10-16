"""Conditional target-sensor forecasting with past known ESA telecommands.

Ablation against the *same* Ridge autoregressor without commands. Signals
have meaning only as anonymized command pulses, not verified physical modes.
"""
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge


class TargetRidgeWithCommands:
    """Forecast target x_t from last 8 target values and observed commands <=t.

    Fit only on confirmed nominal training windows. Commands are model inputs,
    NEVER training targets, and there is no use of labels during test fitting.
    """
    def __init__(self,alpha=10.):
        self.alpha=float(alpha)

    def _xy(self,windows):
        w=np.asarray(windows,float)
        if w.ndim!=3 or w.shape[1]<10 or w.shape[2]<2:
            raise ValueError('Expected (windows, >=10 lags, target + command features)')
        target_lags=w[:,-9:-1,0]
        cmds=w[:,:,1:]
        f=np.concatenate([target_lags,
            cmds[:,-1,:],
            cmds[:,-4:,:].sum(axis=1),
            cmds[:,-8:,:].sum(axis=1)],axis=1)
        return f,w[:,-1,0]

    def fit(self,windows):
        x,y=self._xy(windows)
        self.scaler=StandardScaler().fit(x)
        self.reg=Ridge(alpha=self.alpha).fit(self.scaler.transform(x),y)
        residual=y-self.reg.predict(self.scaler.transform(x))
        self.residual_scale=max(float(np.std(residual)),1e-5)
        return self

    def score(self,windows):
        x,y=self._xy(windows)
        return np.abs(y-self.reg.predict(self.scaler.transform(x)))/self.residual_scale
