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
        self.ndays = cfg.getAttributeDefault('ndays', 21)
        self.lag = cfg.getAttributeDefault('lag', 0)
        self.cap = dr.getData('cap')
        self.close = dr.getData('adj_close')
        self.status = dr.getData('status')

        self.risk_free_return = 0


    def generate(self, di):
        start_di = di - self.delay - self.lag - self.ndays + 1
        end_di = di - self.delay - self.lag + 1

        valid_idx = self.valid[di] #& self.valid[di-1] & (self.volume[di - 1] > 0) & (self.volume[di] > 0)
        #停牌后复牌的股票不要
        valid_idx = valid_idx & ~((self.status[di - 1] == 0) & (self.status[di] == 1))

        returns = self.close[start_di: end_di, valid_idx] / self.close[start_di -1: end_di - 1, valid_idx]

        #std_returns = Oputil.std(returns)
        close = self.close[di - self.delay - 21 + 1 : di - self.delay + 1, valid_idx]
        std_close = cuiyfOp.std(self.close[di - self.delay - 5 + 1: di - self.delay + 1, valid_idx])
        std_close_lag = cuiyfOp.std(self.close[di - self.delay - 63 - 5 + 1: di - self.delay - 5 + 1, valid_idx])

        close_rank = cuiyfOp.rank(cuiyfOp.zscore(close, axis = 0)[-1])

        alpha = -(std_close/std_close_lag) * close_rank
        self.alpha[valid_idx] = alpha
        

    """
    def checkpointSave(self, fh):
        pickle.dump(self.pre_alpha, fh)

    def checkpointLoad(self, fh):
        self.pre_alpha = pickle.load(fh)
    """