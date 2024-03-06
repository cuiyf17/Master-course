# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
from alphasim import AlphaBase
from alphasim import DataRegistry as dr
from alphasim import Oputil

# 将其他路径添加到sys.path中
sys.path.append(os.path.abspath('/home/cuiyf/myalphasim/'))
from cuiyf_op import cuiyfOp as cuiyfOp


class AlphaBarraResidual(AlphaBase):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.ndays = cfg.getAttributeDefault('ndays', 252)
        self.halflife = cfg.getAttributeDefault('halflife', 63)
        self.close = dr.getData('adj_close')
        self.volume = dr.getData('adj_volume')
        self.returns = dr.getData('returns')
        self.cap = dr.getData('cap')
        self.status = dr.getData('status')
        self.risk_free_return = 0

    def generate(self, di):
        start_di = di - self.delay - self.ndays + 1
        end_di = di - self.delay + 1

        valid_idx = self.valid[di] #& self.valid[di-1] & (self.volume[di - 1] > 0) & (self.volume[di] > 0)
        #停牌后复牌的股票不要
        valid_idx = valid_idx & ~((self.status[di - 1] == 0) & (self.status[di] == 1))
        
        returns = (self.close[start_di : end_di, valid_idx] / self.close[start_di - 1 : end_di - 1, valid_idx] - 1) - self.risk_free_return
        cap_weighted_excess_returns = (returns * self.cap[start_di : end_di, valid_idx]).mean(axis=1, keepdims=True) - self.risk_free_return


        weights = np.sqrt(np.exp(-np.log(2)*(np.arange(self.ndays-1, -1, -1)).reshape(-1, 1)/self.halflife))
        ewt_cap_weighted_returns = cap_weighted_excess_returns*weights
        ewt_returns = returns*weights

        residual = cuiyfOp.tsLinearRegression(ewt_cap_weighted_returns, ewt_returns, output="residual")
        #residual = cuiyfOp.zscore(residual)[-1,:]
        residual = cuiyfOp.tsEMA(residual)


        returns = self.close[di - self.delay, valid_idx] / self.close[di - self.delay - 5, valid_idx]
        my_groups = cuiyfOp.group_split(returns, 5)

        residual = cuiyfOp.groupNeutralize(residual, my_groups)


        self.alpha[valid_idx] = -residual
