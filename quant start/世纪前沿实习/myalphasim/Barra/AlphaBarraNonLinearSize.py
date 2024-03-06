# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
from alphasim import AlphaBase
from alphasim import DataRegistry as dr
from alphasim import Oputil

# 将其他路径添加到sys.path中
sys.path.append(os.path.abspath('/home/cuiyf/myalphasim/'))
from cuiyf_op.neutralizeVector import neutralizeVector as neutralizeVector
from cuiyf_op.truncate import truncate as truncate

class AlphaBarraSize(AlphaBase):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.status = dr.getData('status')
        self.cap = dr.getData('cap')

    def generate(self, di):
        valid_idx = self.valid[di] #& self.valid[di-1] & (self.volume[di - 1] > 0) & (self.volume[di] > 0)
        #停牌后复牌的股票不要
        valid_idx = valid_idx & ~((self.status[di - 1] == 0) & (self.status[di] == 1))

        cap = self.cap[di - self.delay, valid_idx]
        #cap /= np.nanmax(cap)
        cubed_cap = cap**3

        alpha = neutralizeVector(input = cubed_cap, vector = cap, method = "Schmidt")
        truncate(alpha)
        alpha = (alpha - np.nanmean(alpha))/(np.nanstd(alpha) + 1e-12)

        self.alpha[valid_idx] = alpha
