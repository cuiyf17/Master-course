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
        self.ndays = cfg.getAttributeDefault('ndays', 126)
        self.lag = cfg.getAttributeDefault('lag', 21)
        self.close = dr.getData('adj_close')
        self.volume = dr.getData('adj_volume')
        self.amount = dr.getData('amount')
        self.status = dr.getData('status')
        self.risk_free_return = 0

        self.pre_alpha = None

    def generate(self, di):
        start_di = di - self.delay - self.lag - self.ndays + 1
        end_di = di - self.delay - self.lag + 1

        valid_idx = self.valid[di] #& self.valid[di-1] & (self.volume[di - 1] > 0) & (self.volume[di] > 0)
        #停牌后复牌的股票不要
        valid_idx = valid_idx & ~((self.status[di - 1] == 0) & (self.status[di] == 1))

        meanamount1 = cuiyfOp.mean(self.amount[start_di + self.ndays: end_di + self.lag, valid_idx])
        meanamount2 = cuiyfOp.mean(self.amount[start_di: end_di, valid_idx])
        stdamount = cuiyfOp.std(self.amount[start_di + self.ndays: end_di + self.lag, valid_idx])
        #cumvolume = cuiyfOp.sum(self.volume[start_di + self.ndays: end_di + self.lag, valid_idx])
        #stdvolume = cuiyfOp.std(self.volume[start_di + self.ndays: end_di + self.lag, valid_idx])
        returns = self.close[di - self.delay - self.lag, valid_idx]/self.close[di - self.delay - self.lag - self.ndays, valid_idx] - 1
        r = 0.75
        popularity = cuiyfOp.rank((meanamount1/meanamount2)/stdamount)
        #Wpopularity = np.where(popularity <0.5, np.nan, popularity)
        # truncate 极端值
        #Oputil.rank(returns)
        #returns = cuiyfOp.quantileTruncate(returns, 0.2, set_nan = True)
        #ret_idx = np.argsort(returns)
        #returns[ret_idx[:4]] = np.nan
        #returns[ret_idx[-1]] = np.nan
        alpha = returns/(popularity+1e-5)
        group = cuiyfOp.group_split2(alpha, 10)
        #group_mean = cuiyfOp.mean(group)
        #Oputil.groupRank(returns, group)
        #alpha = (group - group_mean)/(popularity+1e-5)
        alpha = group
        
        if(self.pre_alpha is None):
            self.alpha[valid_idx] = alpha
            self.pre_alpha = self.alpha.copy()
        else:
            #bcorr = cuiyfOp.corr(alpha, self.pre_alpha[valid_idx])
            mask1 = ~np.isnan(self.pre_alpha[valid_idx])
            mask2 = ~np.isnan(self.pre_alpha)
            if((di - self.delay)%1 == 0):
                self.alpha[valid_idx&mask2] = 0*self.pre_alpha[valid_idx&mask2] + 1*alpha[mask1]
                self.alpha[valid_idx&~mask2] = alpha[~mask1]
                self.pre_alpha = self.alpha.copy()
            else:
                self.alpha[valid_idx&mask2] = 1*self.pre_alpha[valid_idx&mask2] + 0*alpha[mask1]
                self.alpha[valid_idx&~mask2] = alpha[~mask1]
                self.pre_alpha = self.alpha.copy()

    def checkpointSave(self, fh):
        pickle.dump(self.pre_alpha, fh)

    def checkpointLoad(self, fh):
        self.pre_alpha = pickle.load(fh)