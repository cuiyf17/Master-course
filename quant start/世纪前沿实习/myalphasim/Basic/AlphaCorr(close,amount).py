# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
from alphasim import AlphaBase
from alphasim import DataRegistry as dr
from alphasim import Oputil

# 将其他路径添加到sys.path中
sys.path.append(os.path.abspath('/home/cuiyf/myalphasim/'))



class Alpha5dr(AlphaBase):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.ndays = cfg.getAttributeDefault('ndays', 5)
        self.status = dr.getData('status')
        self.amount = dr.getData('amount')
        self.close = dr.getData('adj_close')
        self.risk_free_return = 0

    def generate(self, di):
        start_di = di - self.delay - self.ndays + 1
        end_di = di - self.delay + 1

        valid_idx = self.valid[di] #& self.valid[di-1] & (self.volume[di - 1] > 0) & (self.volume[di] > 0)
        #停牌后复牌的股票不要
        valid_idx = valid_idx & ~((self.status[di - 1] == 0) & (self.status[di] == 1))

        close = self.close[start_di: end_di, valid_idx]
        amount = self.amount[start_di: end_di, valid_idx]
        
        alpha = -Oputil.corr(close, amount, axis=0)

        self.alpha[valid_idx] = alpha
