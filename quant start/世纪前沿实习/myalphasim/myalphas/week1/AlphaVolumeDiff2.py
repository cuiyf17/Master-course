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
import cuiyf_op.cuiyfOp as OP


class Alpha(AlphaBase):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.ndays = cfg.getAttributeDefault('ndays', 22+3)
        self.lag = cfg.getAttributeDefault('lag', 0)
        self.cap = dr.getData('cap')
        self.close = dr.getData('adj_close')
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

        days = 5
        volume = self.volume[start_di : end_di, valid_idx]
        alpha = self.volume[start_di : end_di, valid_idx] - 2*self.volume[start_di - days: end_di - days, valid_idx] + self.volume[start_di - 2*days: end_di - 2*days, valid_idx]
        alpha = -OP.zscore(alpha)[-1]
        #alpha = cuiyfOp.tsEMA(alpha, alpha=0.8)

        g_returns = self.close[di-self.delay, valid_idx]/self.close[di-self.delay-5, valid_idx]-1
        group1 = OP.group_split2(g_returns, 5)
        alpha = OP.groupNeutralize(alpha, group1)
        # alpha = OP.rank(alpha)
        alpha = OP.power(alpha, 0.5)
        

        self.alpha[valid_idx] = alpha

    '''
    def checkpointSave(self, fh):
        pickle.dump(self.tszscore, fh)

    def checkpointLoad(self, fh):
        self.tszscore = pickle.load(fh)
    '''