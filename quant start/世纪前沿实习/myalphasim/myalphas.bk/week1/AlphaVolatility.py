# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
import pickle
from alphasim import AlphaBase
from alphasim import DataRegistry as dr
from alphasim import Oputil

# 将其他路径添加到sys.path中
sys.path.append(os.path.abspath('/home/cuiyf/myalphasim/'))
import cuiyf_op.cuiyfOp as cuiyfOp


class Alpha(AlphaBase):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.ndays = cfg.getAttributeDefault('ndays', 22)
        self.lag = cfg.getAttributeDefault('lag', 0)
        self.cap = dr.getData('cap')
        self.open = dr.getData('adj_open')
        self.close = dr.getData('adj_close')
        self.high = dr.getData('adj_high')
        self.low = dr.getData('adj_low')
        self.vwap = dr.getData('adj_vwap')
        self.volume = dr.getData('adj_volume')
        self.returns = dr.getData('returns')
        self.status = dr.getData('status')
        self.risk_free_return = 0

        self.tszscore = None

    def generate(self, di):
        start_di = di - self.delay - self.lag - self.ndays + 1
        end_di = di - self.delay - self.lag + 1

        valid_idx = self.valid[di] #& self.valid[di-1] & (self.volume[di - 1] > 0) & (self.volume[di] > 0)
        #停牌后复牌的股票不要
        valid_idx = valid_idx & ~((self.status[di - 1] == 0) & (self.status[di] == 1))

        returns = self.returns[start_di : end_di, valid_idx]
        std_returns = Oputil.std(returns)

        alpha = std_returns * (self.high[di - self.delay, valid_idx] - self.low[di - self.delay, valid_idx])/self.vwap[di - self.delay, valid_idx]

        self.alpha[valid_idx] = -alpha

    '''
    def checkpointSave(self, fh):
        pickle.dump(self.tszscore, fh)

    def checkpointLoad(self, fh):
        self.tszscore = pickle.load(fh)
    '''
    #截面相似度，截面condition