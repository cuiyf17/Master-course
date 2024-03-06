# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
import pickle
from alphasim import AlphaBase
from alphasim import DataRegistry as dr
from alphasim import Oputil
from alphasim import Universe as uv


# 将其他路径添加到sys.path中
sys.path.append(os.path.abspath('/home/cuiyf/myalphasim/'))
import cuiyf_op.cuiyfOp as OP
from sklearn.decomposition import PCA

import warnings
warnings.filterwarnings('ignore')

class Alpha(AlphaBase):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.ndays = cfg.getAttributeDefault('ndays', 126)
        self.lag = cfg.getAttributeDefault('lag', 0)

        self.status = dr.getData('status')
        self.cap = dr.getData('cap')
        self.negcap = dr.getData('negcap')
        self.amount = dr.getData('amount')

        self.open = dr.getData('open')
        self.close = dr.getData('close')
        self.high = dr.getData('high')
        self.low = dr.getData('low')
        self.vwap = dr.getData('vwap')
        self.volume = dr.getData('volume')

        self.adj_open = dr.getData('adj_open')
        self.adj_close = dr.getData('adj_close')
        self.adj_high = dr.getData('adj_high')
        self.adj_low = dr.getData('adj_low')
        self.adj_vwap = dr.getData('adj_vwap')
        self.adj_volume = dr.getData('adj_volume')
        
        self.subindustry = dr.getData('subindustry')
        self.industry = dr.getData('industry')
        self.sector = dr.getData('sector')
        self.family = dr.getData('family')
        self.country = dr.getData('country')
        self.hs300 = dr.getData('index.hs300')
        self.zz500 = dr.getData('index.zz500')
        self.zz1000 = dr.getData('index.zz1000')

        self.ihigh = dr.getData('interval5m.high')
        self.ilow = dr.getData('interval5m.low')	
        self.iclose = dr.getData('interval5m.close')
        self.iopen = dr.getData('interval5m.open')

        self.total_asset = dr.getData("WindBalancesheet_Q.TOT_LIAB_SHRHLDR_EQY") #总资产
        self.total_income = dr.getData("WindIncome_Q.TOT_OPER_REV") #营业总收入

        self.buy_exlarge = dr.getData("MoneyFlow.buyValueExlarge")
        self.buy_exlarge_act = dr.getData("MoneyFlow.buyValueExlargeAct")
        self.sell_exlarge = dr.getData("MoneyFlow.sellValueExlarge")
        self.sell_exlarge_act = dr.getData("MoneyFlow.sellValueExlargeAct")
        self.buy_exlarge_act = dr.getData("MoneyFlow.buyValueExlargeAct")
        self.buy_large = dr.getData("MoneyFlow.buyValueLarge")
        self.buy_large_act = dr.getData("MoneyFlow.buyValueLargeAct")
        self.sell_large = dr.getData("MoneyFlow.sellValueLarge")
        self.sell_large_act = dr.getData("MoneyFlow.sellValueLargeAct")
        self.buy_medium = dr.getData("MoneyFlow.buyValueMed")
        self.buy_medium_act = dr.getData("MoneyFlow.buyValueMedAct")
        self.sell_medium = dr.getData("MoneyFlow.sellValueMed")
        self.sell_medium_act = dr.getData("MoneyFlow.sellValueMedAct")
        self.buy_small = dr.getData("MoneyFlow.buyValueSmall")
        self.buy_small_act = dr.getData("MoneyFlow.buyValueSmallAct")
        self.sell_small = dr.getData("MoneyFlow.sellValueSmall")
        self.sell_small_act = dr.getData("MoneyFlow.sellValueSmallAct")

        self.est_dt = dr.getData("WIND_AShareEarningEst.EST_DT")
        self.analyst_name = dr.getData("WIND_AShareEarningEst.ANALYST_NAME")
        # self.analyst_id = dr.getData("WIND_AShareEarningEst.ANALYST_CODE")
        self.research_inst_name = dr.getData("WIND_AShareEarningEst.RESEARCH_INST_NAME")
        self.reprting_period = dr.getData("WIND_AShareEarningEst.REPORTING_PERIOD")
        self.est_eps_diluted = dr.getData("WIND_AShareEarningEst.EST_EPS_DILUTED")

        self.offset = dr.getData("WIND_AShareEarningEst.offset")

        self.risk_free_return = 0
        self.pre_alpha = None #不可删除，用于动量更新控制换手率
        self.pre_alphas = None #不可删除，用于止盈止损
        self.pre_alphas_zscore = None #不可删除，用于zscore控制开仓平仓
        self.holding_days = None

    def generate(self, di):
        num_stocks = len(self.close[di-self.delay])
        start_di = di - self.delay - self.lag - self.ndays + 1
        end_di = di - self.delay - self.lag + 1

        valid_idx = self.valid[di] #& self.valid[di-1] & (self.volume[di - 1] > 0) & (self.volume[di] > 0)
        # 停牌后复牌的股票不要
        valid_idx = valid_idx & ~((self.status[di-self.delay - 1] == 0) & (self.status[di-self.delay] == 1))
        # 前一天涨跌停的股票不要
        # mask1 = (np.abs(self.close[di-self.delay]/self.close[di-self.delay-1]-1) > 0.198) & (self.family[di-self.delay]==2)
        # mask2 = (np.abs(self.close[di-self.delay]/self.close[di-self.delay-1]-1) > 0.099) & (self.family[di-self.delay]!=2)
        # valid_idx = valid_idx & ~mask1 & ~mask2

        # 计算涨跌停股票
        returns = self.adj_close[start_di:end_di]/self.adj_close[start_di-1:end_di-1]-1
        stop_board = np.any(np.abs(returns) > 0.095, axis=0)
        
        start = self.offset[start_di:end_di, :, 0, 0]
        stop = self.offset[start_di:end_di, :, :, 0].sum(axis=2)

        est_idxs_per_stock = [[] for _ in range(num_stocks)]
        tmp_idxs = [list(range(start[i, ii], stop[i, ii])) for i in range(start.shape[0]) for ii in range(len(est_idxs_per_stock))]
        est_idxs_per_stock = [est_idxs_per_stock[ii] + tmp_idxs[ii] for ii in range(len(est_idxs_per_stock))]

        all_analyst_names = set()
        est_analyst_names = [set(self.analyst_name[x]) for x in est_idxs_per_stock]
        all_analyst_names = all_analyst_names.union(*est_analyst_names)

        similar_matrix = np.zeros((num_stocks, num_stocks))
        for name in all_analyst_names:
            idxs = [ii for ii in range(len(est_idxs_per_stock)) if name in est_analyst_names[ii]]
            if len(idxs) <= 1:
                continue
            ij_where = np.array([np.array([i, j]) for i in idxs for j in idxs])
            similar_matrix[ij_where[:, 0], ij_where[:, 1]] = similar_matrix[ij_where[:, 0], ij_where[:, 1]] + 1
        print(len(np.where(similar_matrix > 0)[0]))
        ret = self.adj_close[di-self.delay] / self.adj_close[di-self.delay-21] - 1
        ret = ret.reshape((1, -1))

        similar_sum = np.nansum(similar_matrix, axis=1)
        print(np.sum(similar_sum >0))
        alpha = np.nansum(similar_matrix * ret, axis=1)/similar_sum
        # alpha = np.where(similar_sum != 0, alpha, np.nan)


        # zscore开仓平仓
        # alpha = self.zscore_position(alpha, di, openthres = 0.1, holdthres = 0., use_tail = True)
        # alpha = alpha*2
        # 止盈止损
        # alpha = self.corr_stop_profit_loss(alpha, di, lookback = 5, stop_profit = 1, stop_loss = -0.6, excess_zz500 = True, continuous = True)
        # 动量更新
        alpha = alpha[valid_idx]
        self.momentum_update(alpha, di, valid_idx, trade_interval = 1, momentum = 1)
        self.alpha = OP.rank(self.alpha) - 0.5
        self.alpha = OP.power(self.alpha, 3)
        # self.alpha = OP.groupNeutralize(self.alpha, self.country[di-self.delay])
        # self.alpha = OP.group_mean(self.alpha, self.subindustry[di-self.delay])        

    def checkpointSave(self, fh):
        pickle.dump(self.pre_alpha, fh)
        pickle.dump(self.pre_alphas, fh)
        pickle.dump(self.pre_alphas_zscore, fh)
        

    def checkpointLoad(self, fh):
        self.pre_alpha = pickle.load(fh)
        self.pre_alphas = pickle.load(fh)
        self.pre_alphas_zscore = pickle.load(fh)
        
    def zscore_position(self, alpha, di, openthres = 0.6, holdthres = 0.4, use_tail = False):
        tmp = np.zeros((1, self.close[di-1].shape[0]))
        tmp[0] = alpha
        if(self.pre_alphas_zscore is None):
            self.pre_alphas_zscore = tmp
        elif(self.pre_alphas_zscore.shape[0] < 256):
            self.pre_alphas_zscore = np.concatenate((self.pre_alphas_zscore, tmp), axis=0)
        else:
            self.pre_alphas_zscore = np.concatenate((self.pre_alphas_zscore[1:], tmp), axis=0)
        zscore = (alpha - np.nanmean(self.pre_alphas_zscore[:], axis=0))/np.nanstd(self.pre_alphas_zscore[:], axis=0)
        zscore = np.clip(zscore, -3, 3)/3
        if(self.pre_alpha is not None):
            choose_where = ((np.abs(zscore) > openthres) & (np.isnan(self.pre_alpha))) | ((np.abs(zscore) > holdthres) & (~np.isnan(self.pre_alpha)))
        else:
            choose_where = np.abs(zscore) > openthres
        if(use_tail):
            ones = np.where(choose_where, 1, np.nan)
            zscore[choose_where] = zscore[choose_where]/np.nansum(np.abs(zscore[choose_where]))*np.nansum(ones)
            alpha = np.where(choose_where, zscore, np.nan)/alpha.shape[0]
        else:
            alpha = np.where(choose_where, np.sign(zscore), np.nan)/alpha.shape[0]
        
        return alpha
    
    def corr_stop_profit_loss(self, alpha, di, lookback = 5, stop_profit = 0.7, stop_loss = -0.7, excess_zz500 = True, continuous = True):
        tmp = np.zeros((1, self.close[di-1].shape[0]))
        tmp[0] = alpha
        if(self.pre_alphas is None):
            self.pre_alphas = tmp
        elif(self.pre_alphas.shape[0] < lookback):
            self.pre_alphas = np.concatenate((self.pre_alphas, tmp), axis=0)
        else:
            self.pre_alphas = np.concatenate((self.pre_alphas[1:], tmp), axis=0)
        
        nn = self.pre_alphas.shape[0]
        if(excess_zz500):
            zz500_ret = (self.zz500[di-self.delay-nn+1:di-self.delay+1]/self.zz500[di-self.delay-nn:di-self.delay] - 1).reshape((-1, 1))
            ex_ret = self.close[di-self.delay-nn+1:di-self.delay+1]/self.close[di-self.delay-nn:di-self.delay] - 1 - zz500_ret
        else:
            ex_ret = self.close[di-self.delay-nn+1:di-self.delay+1]/self.close[di-self.delay-nn:di-self.delay] - 1
        alpha_momentum = OP.corr(ex_ret, self.pre_alphas[:])
        
        if(continuous):
            coef = np.zeros_like(alpha)+np.nan
            nanwhere = np.where(np.isnan(alpha_momentum))
            positive = np.where(alpha_momentum >= 0)
            negative = np.where(alpha_momentum < 0)
            coef[positive] = 1 - np.minimum(alpha_momentum[positive]/stop_profit, 1)
            coef[negative] = 1 - np.minimum(np.abs(alpha_momentum[negative]/stop_loss), 1)
            coef[nanwhere] = 1
            alpha = alpha * coef
        else:
            coef = np.ones_like(alpha)
            nanwhere = np.where(np.isnan(alpha_momentum))
            positive = np.where(alpha_momentum > stop_profit)
            negative = np.where(alpha_momentum < stop_loss)
            middle = np.where((alpha_momentum <= stop_profit) & (alpha_momentum >= stop_loss))
            coef[positive] = np.nan
            coef[negative] = np.nan
            coef[middle] = 1
            coef[nanwhere] = 1
            alpha = alpha * coef
        
        return alpha
    
    def momentum_update(self, alpha, di, valid_idx, trade_interval = 1, momentum = 0.05):
        if(self.pre_alpha is None):
            self.alpha[valid_idx] = alpha
            self.pre_alpha = self.alpha.copy()
        else:
            mask1 = ~np.isnan(self.pre_alpha[valid_idx])
            mask2 = ~np.isnan(self.pre_alpha)
            if((di - self.delay)%trade_interval == 0):
                self.alpha[valid_idx&mask2] = (1-momentum)*self.pre_alpha[valid_idx&mask2] + momentum*alpha[mask1]
                self.alpha[valid_idx&~mask2] = alpha[~mask1]
                self.pre_alpha = self.alpha.copy()
            else:
                self.alpha[valid_idx&mask2] = 1*self.pre_alpha[valid_idx&mask2] + 0*alpha[mask1]
                self.alpha[valid_idx&~mask2] = alpha[~mask1]
                self.pre_alpha = self.alpha.copy()