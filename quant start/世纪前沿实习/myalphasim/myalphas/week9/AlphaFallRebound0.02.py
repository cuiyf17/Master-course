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

# import warnings
# warnings.filterwarnings('ignore')

class Alpha(AlphaBase):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.ndays = cfg.getAttributeDefault('ndays', 63)
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
        self.adjfactor = dr.getData('adjfactor')
        
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

        self.risk_free_return = 0
        self.pre_alpha = None #不可删除，用于动量更新控制换手率
        self.pre_alphas = None #不可删除，用于止盈止损
        self.pre_alphas_zscore = None #不可删除，用于zscore控制开仓平仓
        self.pre_volatility = None #不可删除，用于波动率控制总仓位大小
        self.holding_days = None #不可删除，用于记录持仓天数
        self.num_triggers = 0
        self.sector_atr = None
        self.start_prices = None
        self.sector_indexs = None

    def generate(self, di):
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

        # 计算行业指数
        if(self.start_prices is None):
            self.start_prices = np.zeros_like(self.close[di-self.delay]) - 1
        where = np.where((self.start_prices == -1) | (np.isnan(self.start_prices)))
        self.start_prices[where] = self.close[di-self.delay, where]
        self.start_prices = self.start_prices*self.adjfactor[di-self.delay]
        indexs = self.close[di-self.delay]/self.start_prices
        sector_index = OP.group_mean(indexs * np.log(self.cap[di-self.delay]), self.sector[di-self.delay])/OP.group_mean(np.log(self.cap[di-self.delay]), self.sector[di-self.delay])
        tmp = np.zeros((1, self.close[di-self.delay].shape[0]))
        tmp[0] = sector_index
        if(self.sector_indexs is None):
            self.sector_indexs = tmp
        elif(self.sector_indexs.shape[0] < 21):
            self.sector_indexs = np.concatenate((self.sector_indexs, tmp), axis=0)
        else:
            self.sector_indexs = np.concatenate((self.sector_indexs[1:], tmp), axis=0)

        # #根据波动率调整大跌和底部反弹的阈值
        # vol1 = self.adj_high[di-self.delay]/self.adj_low[di-self.delay]-1
        # vol2 = np.abs(self.adj_high[di-self.delay]/self.adj_close[di-self.delay-1]-1)
        # vol3 = np.abs(self.adj_low[di-self.delay]/self.adj_close[di-self.delay-1]-1)
        # atr = np.maximum(vol1, vol2, vol3)
        # sector_atr = OP.group_mean(atr * np.log(self.cap[di-self.delay]), self.sector[di-self.delay])/OP.group_mean(np.log(self.cap[di-self.delay]), self.sector[di-self.delay])
        # tmp = np.zeros((1, self.close[di-self.delay].shape[0]))
        # tmp[0] = sector_atr
        # if(self.sector_atr is None):
        #     self.sector_atr = tmp
        # elif(self.sector_atr.shape[0] < 63):
        #     self.sector_atr = np.concatenate((self.sector_atr, tmp), axis=0)
        # else:
        #     self.sector_atr = np.concatenate((self.sector_atr[1:], tmp), axis=0)

        #根据波动率调整大跌和底部反弹的阈值
        zz500_rets = self.zz500[start_di:end_di]/self.zz500[start_di-1:end_di-1]-1
        vol = OP.std(zz500_rets)
        # vol = OP.mean(self.sector_atr)[valid_idx][0]
        adj_coef = 1
        if(vol > 0.02):
            adj_coef = np.sqrt(vol/0.02)
        elif(vol < 0.01):
            adj_coef = np.sqrt(vol/0.01)
        thres_fall = -0.05*adj_coef
        thres_rise = 0.02*adj_coef

        is_reverse = False
        # 确认属于大跌反弹的时点
        zz500_ret = self.zz500[di-self.delay]/self.zz500[di-self.delay-1]-1
        if(zz500_ret > thres_rise):
            lookback = 0
            rise = 0
            while(rise <= thres_rise and lookback < 21):
                lookback += 1
                rise = self.zz500[di-self.delay-lookback]/self.zz500[di-self.delay-lookback-1]-1
            if(lookback > 4):
                zz500_prices = self.zz500[di-self.delay-lookback:di-self.delay]
                zz500_argmax = np.argmax(zz500_prices)
                zz500_max = zz500_prices[zz500_argmax]
                zz500_argmin = np.argmin(zz500_prices)
                zz500_min = zz500_prices[zz500_argmin]
                if((zz500_argmax - zz500_argmin < -2)):
                    fall_step = 1 - zz500_min/zz500_max
                    if(fall_step > thres_fall):
                        is_reverse = True
                        self.num_triggers += 1
        print('num_triggers: ', self.num_triggers)
        
        if(is_reverse):
            rets = self.adj_close[di-self.delay-10+1:di-self.delay+1]/self.adj_close[di-self.delay-10:di-self.delay]-1
            alpha = (rets[-1,:] - OP.mean(rets))/OP.std(rets)*OP.moment(rets, 4)
            group = OP.group_split2(alpha, 10)
            alpha = OP.group_mean(alpha, group)
            self.holding_days = np.zeros_like(self.close[di-self.delay])
        else:
            if(self.pre_alpha is None):
                alpha = np.zeros_like(self.close[di-self.delay])
                self.holding_days = np.zeros_like(self.close[di-self.delay])
            else:
                alpha = self.pre_alpha.copy()
                self.holding_days += 1
                longer = np.where(self.holding_days > 21)
                alpha[longer] = 0
                self.holding_days[longer] = 0

        
            
        # zscore开仓平仓
        # alpha = self.zscore_position(alpha, di, openthres = 0.3, holdthres = 0.1, use_tail = True)
        # 止盈止损
        alpha = self.corr_stop_profit_loss(alpha, di, lookback = 30, stop_profit = 0.6, stop_loss = -0.8, excess_zz500 = True, continuous = False)
        # 决定总仓位大小
        # alpha = self.get_position_size(alpha, di, whole = False, volatility_pos = True)
        # 动量更新
        alpha = alpha[valid_idx]
        self.momentum_update(alpha, di, valid_idx, trade_interval = 1, momentum = 1)
        self.alpha = OP.rank(self.alpha) - 0.5
        self.alpha = OP.power(self.alpha, 3)
        self.alpha = OP.groupNeutralize(self.alpha, self.country[di-self.delay])
        # self.alpha = OP.group_mean(self.alpha, self.subindustry[di-self.delay])
        

    def checkpointSave(self, fh):
        pickle.dump(self.pre_alpha, fh)
        pickle.dump(self.pre_alphas, fh)
        pickle.dump(self.pre_alphas_zscore, fh)
        pickle.dump(self.pre_volatility, fh)
        pickle.dump(self.holding_days, fh)
        pickle.dump(self.sector_atr, fh)
        pickle.dump(self.start_prices, fh)
        pickle.dump(self.sector_indexs, fh)


    def checkpointLoad(self, fh):
        self.pre_alpha = pickle.load(fh)
        self.pre_alphas = pickle.load(fh)
        self.pre_alphas_zscore = pickle.load(fh)
        self.pre_volatility = pickle.load(fh)
        self.holding_days = pickle.load(fh)
        self.sector_atr = pickle.load(fh)
        self.start_prices = pickle.load(fh)
        self.sector_indexs = pickle.load(fh)

    def zscore_position(self, alpha, di, openthres = 0.6, holdthres = 0.4, use_tail = False):
        num_days = 252
        tmp = np.zeros((1, self.close[di-1].shape[0]))
        tmp[0] = alpha
        if(self.pre_alphas_zscore is None):
            self.pre_alphas_zscore = tmp
        elif(self.pre_alphas_zscore.shape[0] < num_days):
            self.pre_alphas_zscore = np.concatenate((self.pre_alphas_zscore, tmp), axis=0)
        else:
            self.pre_alphas_zscore = np.concatenate((self.pre_alphas_zscore[1:], tmp), axis=0)
        
        zscore = (alpha - np.nanmean(self.pre_alphas_zscore[:], axis=0))/np.nanstd(self.pre_alphas_zscore[:], axis=0)
        zscore = np.clip(zscore, -3, 3)/3
        
        if(self.pre_alpha is not None):
            choose_where = ((np.abs(zscore) > openthres) & (np.isnan(self.pre_alpha))) | ((np.abs(zscore) > holdthres) & (~np.isnan(self.pre_alpha)))
        else:
            choose_where = np.abs(zscore) > openthres
        
        num_stocks = self.valid[di].sum()
        if(use_tail):
            ones = np.where(choose_where, 1, np.nan)
            zscore[choose_where] = zscore[choose_where]/np.nansum(np.abs(zscore[choose_where]))*np.nansum(ones)
            alpha = np.where(choose_where, zscore, np.nan)/num_stocks
        else:
            alpha = np.where(choose_where, np.sign(zscore), np.nan)/num_stocks
        
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
    
    def get_position_size(self, alpha, di, whole = False, volatility_pos = False):
        num_days = 252
        num_stocks = (~np.isnan(alpha[self.valid[di]])).sum()
        whole_stocks = alpha[self.valid[di]].shape[0]
        if(whole):
            whole_stocks = alpha[self.valid[di]].shape[0]
        elif(volatility_pos):
            tmp = np.zeros((1, self.close[di-1].shape[0]))
            volatility = np.maximum(self.adj_high[di-self.delay]/self.adj_low[di-self.delay] - 1, np.abs(self.adj_high[di-self.delay]/self.adj_close[di-self.delay-1] - 1), np.abs(self.adj_low[di-self.delay]/self.adj_close[di-self.delay-1] - 1))
            tmp = np.zeros((1, self.close[di-1].shape[0]))
            tmp[0] = volatility
            if(self.pre_volatility is None):
                self.pre_volatility = tmp
            elif(self.pre_volatility.shape[0] < num_days):
                self.pre_volatility = np.concatenate((self.pre_volatility, tmp), axis=0)
            else:
                self.pre_volatility = np.concatenate((self.pre_volatility[1:], tmp), axis=0)

            vol_zscore = (self.pre_volatility[-1] - OP.mean(self.pre_volatility))/OP.std(self.pre_volatility)
            avg_vol_zscore = np.nansum(self.amount[di-self.delay] * vol_zscore)/np.nansum(self.amount[di-self.delay])
            avg_vol_zscore = avg_vol_zscore/3
            num_stocks = np.maximum(0, whole_stocks - (whole_stocks) * (np.max(avg_vol_zscore, 0)))

        alphasum = np.nansum(np.abs(alpha))
        alpha = alpha/alphasum * num_stocks / whole_stocks
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