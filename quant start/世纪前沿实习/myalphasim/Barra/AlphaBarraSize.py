# -*- coding: utf-8 -*-
import numpy as np
from alphasim import AlphaBase
from alphasim import DataRegistry as dr
from alphasim import Oputil




class AlphaBarraSize(AlphaBase):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.volume = dr.getData('adj_volume')
        self.status = dr.getData('status')
        self.cap = dr.getData('cap')

    def generate(self, di):
        valid_idx = self.valid[di] #& self.valid[di-1] & (self.volume[di - 1] > 0) & (self.volume[di] > 0)
        #停牌后复牌的股票不要
        valid_idx = valid_idx & ~((self.status[di - 1] == 0) & (self.status[di] == 1))
        
        log_cap = np.log(self.cap[di - self.delay, valid_idx])

        self.alpha[valid_idx] = -log_cap
