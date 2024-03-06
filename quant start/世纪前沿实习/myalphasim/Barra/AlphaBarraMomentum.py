# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
from alphasim import AlphaBase
from alphasim import DataRegistry as dr
from alphasim import Oputil

# 将其他路径添加到sys.path中
sys.path.append(os.path.abspath('/home/cuiyf/myalphasim/'))


class AlphaBarraMomentum(AlphaBase):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.ndays = cfg.getAttributeDefault('ndays', 504)
        self.halflife = cfg.getAttributeDefault('halflife', 126)
        self.lag = cfg.getAttributeDefault('lag', 21)
        self.close = dr.getData('adj_close')
        self.volume = dr.getData('adj_volume')
        self.status = dr.getData('status')
        self.risk_free_return = 0

    def generate(self, di):
        start_di = di - self.delay - self.lag - self.ndays + 1
        end_di = di - self.delay - self.lag + 1

        valid_idx = self.valid[di] #& self.valid[di-1] & (self.volume[di - 1] > 0) & (self.volume[di] > 0)
        #停牌后复牌的股票不要
        valid_idx = valid_idx & ~((self.status[di - 1] == 0) & (self.status[di] == 1))
        
        log_returns = self.close[start_di : end_di, valid_idx] / self.close[start_di - 1 : end_di - 1, valid_idx] - 1
        log_returns = np.log(1 + log_returns)
        log_risk_free_return = np.log(1 + self.risk_free_return)

        weights = np.exp(-np.log(2)*(np.arange(self.ndays-1, -1, -1)).reshape(-1, 1)/self.halflife)
        RSTR = Oputil.sum(log_returns * weights, axis=0)

        self.alpha[valid_idx] = RSTR
