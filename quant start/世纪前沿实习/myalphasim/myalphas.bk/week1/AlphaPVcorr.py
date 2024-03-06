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


class AlphaVwapClose(AlphaBase):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.ndays = cfg.getAttributeDefault('ndays', 5)
        self.close = dr.getData('adj_close')
        self.vwap = dr.getData('adj_vwap')
        self.volume = dr.getData('adj_volume')
        self.status = dr.getData('status')
        self.stdret20 = dr.getData('stdret30')
        self.risk_free_return = 0

        self.pre_alpha = None

    def generate(self, di):
        start_di = di - self.delay - self.ndays + 1
        end_di = di - self.delay + 1

        valid_idx = self.valid[di] #& self.valid[di-1] & (self.volume[di - 1] > 0) & (self.volume[di] > 0)
        #停牌后复牌的股票不要
        valid_idx = valid_idx & ~((self.status[di - 1] == 0) & (self.status[di] == 1))

        price = self.vwap[start_di : end_di, valid_idx]
        volume = self.volume[start_di : end_di, valid_idx]

        alpha = Oputil.corr(price, volume)
        if(self.pre_alpha is None):
            tmp = np.zeros((1, self.close[di].shape[0]))
            tmp[0, valid_idx] = alpha
            self.pre_alpha = tmp
        elif(self.pre_alpha.shape[0] < self.ndays):
            tmp = np.zeros((1, self.close[di].shape[0]))
            tmp[0, valid_idx] = alpha
            self.pre_alpha = np.concatenate([self.pre_alpha, tmp], axis=0)
        else:
            tmp = np.zeros((1, self.close[di].shape[0]))
            tmp[0, valid_idx] = alpha
            self.pre_alpha = np.concatenate([self.pre_alpha[1:], tmp], axis=0)

        price_volatility = self.stdret20[di, valid_idx]
        Oputil.rank(price_volatility)

        alpha = -(alpha - self.pre_alpha[0, valid_idx]) * price_volatility

        
        self.alpha[valid_idx] = alpha



    def checkpointSave(self, fh):
        pickle.dump(self.pre_alpha, fh)

    def checkpointLoad(self, fh):
        self.pre_alpha = pickle.load(fh)
