# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
from alphasim import AlphaBase
from alphasim import DataRegistry as dr
from alphasim import Oputil

# 将其他路径添加到sys.path中
sys.path.append(os.path.abspath('/home/cuiyf/myalphasim/'))



class AlphaBarraLiquidity(AlphaBase):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.min_ndays = cfg.getAttributeDefault('min_ndays', 21)
        self.volume = dr.getData('volume')
        self.close = dr.getData('close')
        self.negcap = dr.getData('negcap')
        self.status = dr.getData('status')
        self.risk_free_return = 0

    def generate(self, di):
        num_months = [1, 3, 12]
        
        valid_idx = self.valid[di] #& self.valid[di-1] & (self.volume[di - 1] > 0) & (self.volume[di] > 0)
        #停牌后复牌的股票不要
        valid_idx = valid_idx & ~((self.status[di - 1] == 0) & (self.status[di] == 1))

        alphas = []
        for coef in num_months:
            ndays = self.min_ndays * coef
            start_di = di - self.delay - ndays + 1
            end_di =  di - self.delay + 1
            sharesout = self.negcap[start_di : end_di, valid_idx]/self.close[start_di : end_di, valid_idx]
            volume = self.volume[start_di : end_di, valid_idx]
            STO = Oputil.sum(np.log((volume/sharesout) + 1e-12), axis = 0)/coef
            alphas.append(STO)

        self.alpha[valid_idx] = alphas[0]*0.35 + alphas[1]*0.35 + alphas[2]*0.3
