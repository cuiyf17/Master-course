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
        self.ndays = cfg.getAttributeDefault('ndays', 63)
        self.lag = cfg.getAttributeDefault('lag', 21)
        self.cap = dr.getData('cap')
        self.close = dr.getData('adj_close')
        self.status = dr.getData('status')

        self.risk_free_return = 0
        self.pre_alpha = None

    def generate(self, di):
        start_di = di - self.delay - self.lag - self.ndays + 1
        end_di = di - self.delay - self.lag + 1

        valid_idx = self.valid[di] #& self.valid[di-1] & (self.volume[di - 1] > 0) & (self.volume[di] > 0)
        #停牌后复牌的股票不要
        valid_idx = valid_idx & ~((self.status[di - 1] == 0) & (self.status[di] == 1))

        #returns = self.close[di - self.delay - 21 + 1 : di - self.delay + 1, valid_idx]/self.close[di - self.delay - 21 : di - self.delay, valid_idx]
        returns = self.close[di - self.delay, valid_idx]/self.close[di - self.delay - 21, valid_idx]

        nday = 5
        #close_diff2 = self.close[di - self.delay, valid_idx] - self.close[di - self.delay - 10, valid_idx] - self.close[di - self.delay - 5, valid_idx] + self.close[di - self.delay - 5 - 10, valid_idx]
        nd_close1 = self.close[di - self.delay - 5 + 1 : di - self.delay + 1, valid_idx]
        nd_num1 = np.arange(nd_close1.shape[0]).repeat(nd_close1.shape[1]).reshape(nd_close1.shape).astype(np.float32)
        nd_close2 = self.close[di - self.delay - 5 - 10 + 1 : di - self.delay - 5 + 1, valid_idx]
        nd_num2 = np.arange(nd_close2.shape[0]).repeat(nd_close2.shape[1]).reshape(nd_close2.shape).astype(np.float32)
        close_diff2 = cuiyfOp.corr(nd_close1, nd_num1)*np.sqrt(cuiyfOp.std(nd_close1)) - cuiyfOp.corr(nd_close2, nd_num2)*np.sqrt(cuiyfOp.std(nd_close2))
        
        #volatility = cuiyfOp.std(returns)

        mygroup = cuiyfOp.group_split2(returns, 5)

        alpha = -close_diff2
        alpha = cuiyfOp.groupNeutralize(alpha, mygroup)
        

        self.alpha[valid_idx] = alpha 

    def checkpointSave(self, fh):
        pickle.dump(self.pre_alpha, fh)

    def checkpointLoad(self, fh):
        self.pre_alpha = pickle.load(fh)
        