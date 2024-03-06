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
from cuiyf_op.tsCorr import tsLinearRegression as tsLR


class AlphaBarraResidualVolatility(AlphaBase):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.ndays = cfg.getAttributeDefault('ndays', 252)
        self.nmonths = cfg.getAttributeDefault('nmonths', 12)
        self.halflife1 = cfg.getAttributeDefault('halflife1', 42)
        self.halflife2 = cfg.getAttributeDefault('halflife2', 63)
        self.close = dr.getData('adj_close')
        self.cap = dr.getData('cap')
        self.status = dr.getData('status')
        self.risk_free_return = 0
        self.pre_alpha = []

    def generate(self, di):
        valid_idx = self.valid[di] #& self.valid[di-1] & (self.volume[di - 1] > 0) & (self.volume[di] > 0)
        #停牌后复牌的股票不要
        valid_idx = valid_idx & ~((self.status[di - 1] == 0) & (self.status[di] == 1))

        start_di = di - self.delay - self.ndays + 1
        end_di = di - self.delay + 1
        excess_return = self.close[start_di : end_di, valid_idx] / self.close[start_di - 1 : end_di - 1, valid_idx] - 1 - self.risk_free_return
        std_return = Oputil.std(excess_return, axis=0)
        if(len(self.pre_alpha) == 0):
            self.pre_alpha = np.zeros((1, self.close[di].shape[0])) + np.nan
            self.pre_alpha[0, valid_idx] = std_return
        elif(len(self.pre_alpha) < self.ndays):
            tmp_alpha = np.zeros_like(self.close[di]) + np.nan
            tmp_alpha[valid_idx] = std_return
            self.pre_alpha = np.vstack((self.pre_alpha, tmp_alpha))
        else:
            tmp_alpha = np.zeros_like(self.close[di]) + np.nan
            tmp_alpha[valid_idx] = std_return
            self.pre_alpha = self.pre_alpha[1:, :]
            self.pre_alpha = np.vstack((self.pre_alpha, tmp_alpha))

        weight1 = np.exp(-np.log(2)*np.arange(len(self.pre_alpha)-1, -1, -1).reshape(-1, 1)/self.halflife1)
        DASTD = Oputil.sum(self.pre_alpha[:, valid_idx]*weight1, axis=0)/Oputil.sum(weight1, axis=0)
        
        Z = []
        for T in range(self.nmonths):
            start_di = di - T*21 - self.delay
            end_di = di - self.delay
            end_close = np.log(self.close[end_di, valid_idx])
            start_close = np.log(self.close[start_di - 1, valid_idx])
            Z.append(end_close - start_close)
        Z = np.array(Z)
        z_max = np.max(Z, axis=0)
        z_min = np.min(Z, axis=0)
        CMRA = z_max - z_min

        start_di = di - self.delay - self.ndays + 1
        end_di = di - self.delay + 1
        returns = (self.close[start_di : end_di, valid_idx] / self.close[start_di - 1 : end_di - 1, valid_idx] - 1) - self.risk_free_return
        cap_weighted_excess_returns = (returns * self.cap[start_di : end_di, valid_idx]).mean(axis=1, keepdims=True) - self.risk_free_return
        weights = np.sqrt(np.exp(-np.log(2)*(np.arange(self.ndays-1, -1, -1)).reshape(-1, 1)/self.halflife2))
        ewt_cap_weighted_returns = cap_weighted_excess_returns*weights
        ewt_returns = returns*weights
        residual = tsLR(ewt_cap_weighted_returns, ewt_returns, output="residual")
        HSIGMA = Oputil.std(residual, axis=0)
        

        self.alpha[valid_idx] = DASTD*0.74 + CMRA*0.16 + HSIGMA*0.1
    
def checkpointSave(self, fh):
    pickle.dump(self.pre_alpha, fh)

def checkpointLoad(self, fh):
    self.pre_alpha = pickle.load(fh)
