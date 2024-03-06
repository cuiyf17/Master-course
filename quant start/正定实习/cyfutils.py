import json
import logging
import os
import shutil
import sys
import math
from scipy.stats.stats import skew
from scipy.stats.stats import kurtosis as kurt
import numpy as np
import pandas as pd
from futures import common, ms
from futures_model import db
from cal_ic import reshape_data, reshape_data_acc, cal_ic, get_ic, cal_ic_raw
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge
from multiprocessing import Pool
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm


from scipy.stats.stats import skew
from scipy.stats.stats import kurtosis as kurt

def group_plot(ypred, ytrue, num_group = 10, name = ""):
    pred_mean, pred_std, pred_skew, pred_kurt = ypred.mean(), ypred.std(), skew(ypred), kurt(ypred)
    true_mean, true_std, true_skew, true_kurt = ytrue.mean(), ytrue.std(), skew(ytrue), kurt(ytrue)

    group = pd.qcut(ypred, num_group, np.arange(num_group))#group_split(ytrue, num_group)
    ret_mean, ret_std, ret_sharpe = np.zeros(num_group), np.zeros(num_group), np.zeros(num_group)
    sorted_group = np.unique(group)
    sorted_group.sort()
    for idx, i in enumerate(sorted_group):
        ret_mean[idx] = ytrue[group == i].mean()
        ret_std[idx] = ytrue[group == i].std()
    ret_sharpe = ret_mean / ret_std
    # 创建一个2x2的图形窗口
    fig= plt.figure(figsize=(15, 10))
    # 添加左上角的子图
    ax1 = fig.add_axes([0.1, 0.55, 0.4, 0.35])  # left, bottom, width, height
    # 添加右上角的子图
    ax2 = fig.add_axes([0.55, 0.55, 0.4, 0.35])
    # 添加下方的子图
    ax3 = fig.add_axes([0.1, 0.1, 0.85, 0.35])
    # 在左上角的子图中绘图
    ax1.hist(ypred, bins=100, alpha=0.5, label='pred, mean: {:.4f}, std: {:.4f}, skew: {:.4f}, kurt: {:.4f}'.format(pred_mean, pred_std, pred_skew, pred_kurt))
    ax1.hist(ytrue, bins=100, alpha=0.5, label='true, mean: {:.4f}, std: {:.4f}, skew: {:.4f}, kurt: {:.4f}'.format(true_mean, true_std, true_skew, true_kurt))
    ax1.legend()
    ax1.grid(True)
    ax1.set_title("IC = {:.4f}".format(np.corrcoef(ypred, ytrue)[0, 1]))
    ax1.set_xlabel("Returns value")
    ax1.set_ylabel("Counts")
    # 在右上角的子图中绘图
    ax2.boxplot([ytrue[group == i] for i in sorted_group])
    ax2.grid(True)
    ax2.set_title("Boxplot of Group returns")
    ax2.set_xlabel("Group")
    ax2.set_ylabel("Returns")
    # 在下方的子图中绘图
    ax3.bar(sorted_group, ret_sharpe)
    ax3.grid(True)
    ax3.set_title("Group IR")
    ax3.set_xlabel("Group")
    ax3.set_ylabel("IR")
    fig.suptitle(name)
    fig.show()


def get_preds(model, data, range = ["train", "cv", "test", "2021", "2022"], process_queue = None):
    preds = dict()
    for item in range:
        preds[item] = pd.Series(model.predict(data[item]))
    if(process_queue is not None):
        process_queue.put((model, preds))
    else:
        return preds

def merge_preds(preds_list, common_keys = ["train", "cv", "test", "2021", "2022"]):
    new_preds = dict()
    for key in common_keys:
        new_preds[key] = pd.concat([preds[key] for preds in preds_list], axis = 1).mean(axis = 1)
    return new_preds

def whole_plot(label, results, vcap_df, look_range, groupnames):
    fig = plt.figure(figsize = (20, 15))
    # 添加左上角的子图
    ax1 = fig.add_axes([0, 0.5, 0.4, 0.5])  # left, bottom, width, height
    ax1.axis("off")
    table = ax1.table(cellText = results.values.round(7), rowLabels = results.index, colLabels = results.columns, loc = "center", cellLoc="center", bbox=[0.25, 0, 0.7, 1], edges = "open")
    table.auto_set_font_size(False)  # 设置自动调整字体大小
    # table.set_fontsize(14)  # 设置字体大小
    # table.auto_set_column_width(False)  # 设置列宽

    ax2 = fig.add_axes([0.4+0.025, 0.8, 0.55, 0.2])  # left, bottom, width, height
    ax2.axis("off")
    table = ax2.table(cellText = vcap_df.values.round(7), rowLabels = vcap_df.index, colLabels = vcap_df.columns, loc = "center", cellLoc="center", bbox=[0, 0, 1, 1], edges = "open")
    table.auto_set_font_size(False)  # 设置自动调整字体大小
    table.set_fontsize(12)  # 设置字体大小

    ax3 = fig.add_axes([0.4+0.025, 0.55, 0.15, 0.2])  # left, bottom, width, height
    ax3.set_title("quantile25")
    ax3.plot(look_range, results.loc["quantile25"], "-o", color="blue", label = "y_pred")
    ax3.axhline(y=-0.0003, color="blue", linestyle="--", label = "-3bp")
    ax3.legend(loc="upper left")
    ax4 = ax3.twinx()
    ax4.plot(look_range, [np.percentile(label[period], 25) for period in look_range], "-o", color="red", label = "y_true")
    ax4.legend(loc="lower left")

    ax5 = fig.add_axes([0.4+0.2+0.025, 0.55, 0.15, 0.2])  # left, bottom, width, height
    ax5.set_title("quantile50")
    ax5.plot(look_range, results.loc["quantile50"], "-o", color="blue", label = "y_pred")
    ax5.axhline(y=0, color="blue", linestyle="--", label = "0bp")
    ax5.legend(loc="upper left")
    ax6 = ax5.twinx()
    ax6.plot(look_range, [np.percentile(label[period], 50) for period in look_range], "-o", color="red", label = "y_true")
    ax6.legend(loc="lower left")

    ax7 = fig.add_axes([0.4+0.4+0.025, 0.55, 0.15, 0.2])  # left, bottom, width, height
    ax7.set_title("quantile75")
    ax7.plot(look_range, results.loc["quantile75"], "-o", color="blue", label = "y_pred")
    ax7.axhline(y=0.0003, color="blue", linestyle="--", label = "3bp")
    ax7.legend(loc="upper left")
    ax8 = ax7.twinx()
    ax8.plot(look_range, [np.percentile(label[period], 75) for period in look_range], "-o", color="red", label = "y_true")
    ax8.legend(loc="lower left")
    
    x = np.arange(len(look_range))
    width = 0.2
    # bp_ratio = [[np.sum(preds[period] < -0.0003)/preds[period].shape[0], np.sum((preds[period] < 0)&(preds[period] >= -0.0003))/preds[period].shape[0], np.sum((preds[period] < 0.0003)&(preds[period] >= 0))/preds[period].shape[0], np.sum(preds[period] >= 0.0003)/preds[period].shape[0]] for period in look_range]
    # print(bp_ratio)
    ax9 = fig.add_axes([0, 0, 0.3, 0.45])  # left, bottom, width, height
    ax9.set_title("group IC")
    for i, grpname in enumerate(groupnames):
        bias = i - len(groupnames)/2
        ax9.bar(x+bias*width, [results.loc[grpname+" IC", period] for period in look_range], width, label = grpname)
    ax9.set_xticks(x)
    ax9.set_xticklabels(look_range)
    ax9.legend()

    ax10 = fig.add_axes([0.35, 0, 0.3, 0.45])  # left, bottom, width, height
    ax10.set_title("group R2")
    for i, grpname in enumerate(groupnames):
        bias = i - len(groupnames)/2
        ax10.bar(x+bias*width, [results.loc[grpname+" R2", period] for period in look_range], width, label = grpname)
    ax10.set_xticks(x)
    ax10.set_xticklabels(look_range)
    ax10.legend()

    ax11 = fig.add_axes([0.7, 0, 0.3, 0.45])  # left, bottom, width, height
    ax11.set_title("group Sharp")
    for i, grpname in enumerate(groupnames):
        bias = i - len(groupnames)/2
        ax11.bar(x+bias*width, [results.loc[grpname+" Sharp", period] for period in look_range], width, label = grpname)
    ax11.set_xticks(x)
    ax11.set_xticklabels(look_range)
    ax11.legend()

    fig.show()

def get_rmse(y_pred, y_true):
    y_pre = y_pred.values
    y_tru = y_true.values
    return np.sqrt(mean_squared_error(y_pre, y_tru))

def get_ic(y_pred, y_true):
    y_pre = y_pred.values
    y_tru = y_true.values
    return np.corrcoef(y_pre, y_tru)[0, 1]

def get_rank_ic(y_pred, y_true):
    y_pre = y_pred.values
    y_tru = y_true.values
    return spearmanr(y_pre, y_tru)[0]

def bi_classify_acc(y_pred, y_true):
    y_pre = y_pred.values
    y_tru = y_true.values
    y_pre = np.where(y_pre >= 0, 1, 0)
    y_tru = np.where(y_tru >= 0, 1, 0)
    return np.sum(y_pre == y_tru) / len(y_pre)

def bi_true_positive(y_pred, y_true):
    y_pre = y_pred.values
    y_tru = y_true.values
    y_pre = np.where(y_pre >= 0, 1, 0)
    y_tru = np.where(y_tru >= 0, 1, 0)
    return np.sum((y_pre == 1) & (y_tru == 1)) / np.sum(y_tru == 1)

def bi_true_negative(y_pred, y_true):
    y_pre = y_pred.values
    y_tru = y_true.values
    y_pre = np.where(y_pre >= 0, 1, 0)
    y_tru = np.where(y_tru >= 0, 1, 0)
    return np.sum((y_pre == 0) & (y_tru == 0)) / np.sum(y_tru == 0)

def bi_f1_score(y_pred, y_true):
    y_pre = y_pred.values
    y_tru = y_true.values
    y_pre = np.where(y_pre >= 0, 1, 0)
    y_tru = np.where(y_tru >= 0, 1, 0)
    tp = np.sum((y_pre == 1) & (y_tru == 1))
    fp = np.sum((y_pre == 1) & (y_tru == 0))
    fn = np.sum((y_pre == 0) & (y_tru == 1))
    return 2 * tp / (2 * tp + fp + fn)

def r2_score(y_pred, y_true):
    y_pre = y_pred.values
    y_tru = y_true.values
    return 1 - np.sum((y_pre - y_tru) ** 2) / np.sum((y_tru - y_tru.mean()) ** 2)

def adj_r2_score(y_pred, y_true, num_features):
    y_pre = y_pred.values
    y_tru = y_true.values
    return 1 - (1 - r2_score(y_pred, y_true)) * (len(y_pre) - 1) / (len(y_pre) - num_features - 1)

def quantile(y_pred, num):
    return np.percentile(y_pred, num)

def quant_ic(y_pred, y_true, upper = 10000, lower = 0):
    y_pre = y_pred.values
    y_tru = y_true.values
    # abspre = np.abs(y_pre)
    abspre = y_pre
    where = np.where((abspre <= upper) & (abspre >= lower))
    y_pre = y_pre[where]
    y_tru = y_tru[where]
    return np.corrcoef(y_pre, y_tru)[0, 1]

def quant_r2(y_pred, y_true, upper = 10000, lower = 0):
    y_pre = y_pred.values
    y_tru = y_true.values
    # abspre = np.abs(y_pre)
    abspre = y_pre
    where = np.where((abspre <= upper) & (abspre >= lower))
    y_pre = y_pre[where]
    y_tru = y_tru[where]
    return 1 - np.sum((y_pre - y_tru) ** 2) / np.sum((y_tru - y_tru.mean()) ** 2)

def quant_sharpe(y_pred, y_true, upper = 10000, lower = 0):
    y_pre = y_pred.values
    y_tru = y_true.values
    # abspre = np.abs(y_pre)
    abspre = y_pre
    where = np.where((abspre <= upper) & (abspre >= lower))
    y_pre = y_pre[where]
    y_tru = y_tru[where]
    return np.mean(y_tru) / np.std(y_tru)

def print_pred_score(preds, label, range = ["train", "cv", "test", "2021", "2022"], scores = ["RMSE", "IC", "Rank_IC", "Binary_acc", "Binary_true_positive", "Binary_true_negative", "Binary_F1", "R2", "quantile25", "quantile50", "quantile75", "<-3bp IC", "-3-0bp IC", "0-3bp IC", ">3bp IC", ">5bp IC", ">8bp IC","<-3bp Sharp", "-3-0bp Sharp", "0-3bp Sharp", ">3bp Sharp", ">5bp Sharp", ">8bp Sharp"], group_show = True, is_print = True):
    score_dict = {"RMSE": {"func": get_rmse, "kwargs": {"y_pred": None, "y_true": None}},
                  "IC" : {"func": get_ic, "kwargs": {"y_pred": None, "y_true": None}},
                  "Rank_IC": {"func": get_rank_ic, "kwargs": {"y_pred": None, "y_true": None}},
                  "Binary_acc": {"func": bi_classify_acc, "kwargs": {"y_pred": None, "y_true": None}},
                  "Binary_true_positive": {"func": bi_true_positive, "kwargs": {"y_pred": None, "y_true": None}},
                  "Binary_true_negative": {"func": bi_true_negative, "kwargs": {"y_pred": None, "y_true": None}},
                  "Binary_F1": {"func": bi_f1_score, "kwargs": {"y_pred": None, "y_true": None}},
                  "R2": {"func": r2_score, "kwargs": {"y_pred": None, "y_true": None}},
                #   "adj R2": {"func": adj_r2_score, "kwargs": {"num_features": len(preds["train"].columns)}},
                  "quantile25": {"func": quantile, "kwargs": {"y_pred": None, "num":25}},
                  "quantile50": {"func": quantile, "kwargs": {"y_pred": None, "num":50}},
                  "quantile75": {"func": quantile, "kwargs": {"y_pred": None, "num":75}},
                  "<-3bp IC": {"func": quant_ic, "kwargs": {"y_pred": None, "y_true": None, "upper": -0.0003, "lower": -10000}},
                  "-3-0bp IC": {"func": quant_ic, "kwargs": {"y_pred": None, "y_true": None, "upper": 0, "lower": -0.0003}},
                  "0-3bp IC": {"func": quant_ic, "kwargs": {"y_pred": None, "y_true": None, "upper": 0.0003, "lower": 0}},
                  ">3bp IC": {"func": quant_ic, "kwargs": {"y_pred": None, "y_true": None, "upper": 10000, "lower": 0.0003}},
                  ">5bp IC": {"func": quant_ic, "kwargs": {"y_pred": None, "y_true": None, "upper": 10000, "lower": 0.0005}},
                  ">8bp IC": {"func": quant_ic, "kwargs": {"y_pred": None, "y_true": None, "upper": 10000, "lower": 0.0008}},
                  "<-3bp R2": {"func": quant_r2, "kwargs": {"y_pred": None, "y_true": None, "upper": -0.0003, "lower": -10000}},
                  "-3-0bp R2": {"func": quant_r2, "kwargs": {"y_pred": None, "y_true": None, "upper": 0, "lower": -0.0003}},
                  "0-3bp R2": {"func": quant_r2, "kwargs": {"y_pred": None, "y_true": None, "upper": 0.0003, "lower": 0}},
                  ">3bp R2": {"func": quant_r2, "kwargs": {"y_pred": None, "y_true": None, "upper": 10000, "lower": 0.0003}},
                  "<-3bp Sharp": {"func": quant_sharpe, "kwargs": {"y_pred": None, "y_true": None, "upper": -0.0003, "lower": -10000}},
                  "-3-0bp Sharp": {"func": quant_sharpe, "kwargs": {"y_pred": None, "y_true": None, "upper": 0, "lower": -0.0003}},
                  "0-3bp Sharp": {"func": quant_sharpe, "kwargs": {"y_pred": None, "y_true": None, "upper": 0.0003, "lower": 0}},
                  ">3bp Sharp": {"func": quant_sharpe, "kwargs": {"y_pred": None, "y_true": None, "upper": 10000, "lower": 0.0003}},
                  ">5bp Sharp": {"func": quant_sharpe, "kwargs": {"y_pred": None, "y_true": None, "upper": 10000, "lower": 0.0005}},
                  ">8bp Sharp": {"func": quant_sharpe, "kwargs": {"y_pred": None, "y_true": None, "upper": 10000, "lower": 0.0008}},
                 }

    results = dict()
    for period in range:
        for score in scores:
            if score not in score_dict:
                raise ValueError(f"score {score} must be one of these: {list(score_dict.keys())}")
            if score not in results:
                results[score] = dict()
            # print("func name: ", score_dict[score]["func"].__name__, "kwargs: ", score_dict[score]["kwargs"])
            kwargs = score_dict[score]["kwargs"]
            if "y_pred" in kwargs:
                kwargs["y_pred"] = preds[period]
            if "y_true" in kwargs:
                kwargs["y_true"] = label[period]
            results[score][period] = score_dict[score]["func"](**kwargs)
            
    results = pd.DataFrame(results).T
    if(is_print):
        print(results)
    if group_show:
        pass
    return results

def save_pred_model(preds, save_name, model = None, range = ["train", "cv", "test", "2021", "2022"]):
    if not os.path.exists(f"/data/futures/lmdb/tcf/60s/features/yfcui1/opt/pred/{save_name}"):
        os.mkdir(f"/data/futures/lmdb/tcf/60s/features/yfcui1/opt/pred/{save_name}")
    if(model is not None):
        suffix = range[0] if len(range) == 1 else f"{range[0]}-{range[-1]}"
        model.save_model(f"/data/futures/lmdb/tcf/60s/features/yfcui1/opt/pred/{save_name}/{save_name}_model_{suffix}.json")
    for period in range:
          preds[period].to_pickle(f"/data/futures/lmdb/tcf/60s/features/yfcui1/opt/pred/{save_name}/{save_name}_{period}.pickle")

def get_xgbmodel_config(model, results, save_name, alphalist = None, save = True):
    columns = results.columns
    config = dict()
    config["save_name"] = save_name
    if("train" in columns):
        config["train RMSE"] = results.loc["RMSE"]["train"]
        config["train IC"] = results.loc["IC"]["train"]
        config["train Rank_IC"] = results.loc["Rank_IC"]["train"]
        config["train_bin_acc"] = results.loc["Binary_acc"]["train"]
    if("cv" in columns):
        config["cv RMSE"] = results.loc["RMSE"]["cv"]
        config["cv IC"] = results.loc["IC"]["cv"]
        config["cv Rank_IC"] = results.loc["Rank_IC"]["cv"]
        config["cv_bin_acc"] = results.loc["Binary_acc"]["cv"]
    if("test" in columns):
        config["test RMSE"] = results.loc["RMSE"]["test"]
        config["test IC"] = results.loc["IC"]["test"]
        config["test Rank_IC"] = results.loc["Rank_IC"]["test"]
        config["test_bin_acc"] = results.loc["Binary_acc"]["test"]
    if("2021" in columns):
        config["2021 IC"] = results.loc["IC"]["2021"]
        config["2021 Rank_IC"] = results.loc["Rank_IC"]["2021"]    
        config["2021_bin_acc"] = results.loc["Binary_acc"]["2021"]


    model_config = model.get_params()
    for key in model_config:
        config[key] = model_config[key]
    
    if(alphalist is not None):
        config["alphalist"] = alphalist

    if save:
        with open(f"/data/futures/lmdb/tcf/60s/features/yfcui1/opt/pred/{save_name}/{save_name}.json", "w") as f:
            json.dump(config, f)
    return config

from tqdm import tqdm

def cal_pred_tvr(save_name):
    ## 输入模型名字，得到估计的TO指标
    """
    save_name: 自己的model名字
    predroot: List of cv test 2021的预测值的pickle文件路径
    预测值存在opt_{save_name}_prediction
    """
    predroot = [f"/data/futures/lmdb/tcf/60s/features/yfcui1/opt/pred/{save_name}/{save_name}_cv.pickle",
                f"/data/futures/lmdb/tcf/60s/features/yfcui1/opt/pred/{save_name}/{save_name}_test.pickle",
                f"/data/futures/lmdb/tcf/60s/features/yfcui1/opt/pred/{save_name}/{save_name}_2021.pickle",
                #f"/data/futures/lmdb/tcf/60s/features/yfcui1/opt/pred/{save_name}/{save_name}_2022.pickle"
                ]
    begin, end = "20190101", "20211231"
    dts = [i for i in common.find_dates() if i >= begin and i <= end]
    dts = np.array(dts)
    if not db.find(end, f"opt_{save_name}_prediction"):
        yhat = pd.concat([pd.read_pickle(i) for i in predroot]).values
        f = pd.read_pickle("/data/futures/lmdb/tcf/60s/features/frame.pickle")
        assert np.sum(f > 0) == len(yhat)
        pred = np.repeat(np.nan, len(f))
        pred[f > 0] = yhat
        for date in tqdm(dts):
            f = db.read(date, "opt_frame")
            l = f.shape[0] * f.shape[1]
            predcut = pd.DataFrame(np.reshape(pred[:l], (f.shape[1], f.shape[0])).T, columns=f.columns, index=f.index)
            if predcut.iloc[0].isna().sum():
                predcut.iloc[0] = predcut.iloc[0].fillna(0)
            predcut = predcut.ffill()

            db.write(date, f"opt_{save_name}_prediction", predcut)
            pred = pred[l:]

    
    begin, end = "20190101", "20211231"
    tvr_lst = []
    

    for date in tqdm(dts):
        a = db.read(date,f'opt_{save_name}_prediction')
        tvr_lst.append((a.diff().abs().sum(axis=1)/a.abs().sum(axis=1).shift(-1)).sum())

    return pd.Series(tvr_lst)

def get_alpha_stat(alphaname, status = "stat"):
    assert status in ["stat", "rstat", "vcaprstat"]
    rstat = pd.DataFrame([db.read(date, f"{alphaname}_{status}").rename(date) for date, _ in db.find(key=f"{alphaname}_{status}")])
    rstat = ms.annual_performance(rstat)
    
    return rstat

def plot_pnl(alphaname, status = "rstat", year = "2021"):
    assert status in ["stat", "rstat", "vcaprstat"]
    rstat = pd.DataFrame([db.read(date, f"{alphaname}_{status}").rename(date) for date, _ in db.find(key=f"{alphaname}_{status}") if date <=f"{year}1231"])

    plt.figure(figsize = (20, 7))
    plt.plot(pd.to_datetime(rstat.index), rstat["pnl"].cumsum())
    plt.title(alphaname)
    plt.grid(True)
    plt.show()

def plot_instrument_pnl(alphaname, status="rstat"):
    assert status in ["stat", "rstat", "vcaprstat"]
    rstat = pd.DataFrame([db.read(date, f"{alphaname}_{status}").rename(date) for date, _ in db.find(key=f"{alphaname}_{status}")])
    pnl_columns = [col for col in rstat.columns if col.endswith("_pnl")]

    plt.figure(figsize=(20, 14))
    cmap = cm.get_cmap('nipy_spectral')
    colors = cmap(np.linspace(0, 1, len(pnl_columns)))
    for i, col in enumerate(pnl_columns, start=0):
        plt.plot(pd.to_datetime(rstat.index), rstat[col].cumsum(), color=colors[i], label = col)

    plt.legend()
    plt.show()

def get_rstat(alphaname, status = "rstat"):
    rstat = pd.DataFrame([db.read(date, f"{alphaname}_{status}").rename(date) for date, _ in db.find(key=f"{alphaname}_{status}")])
    return rstat

def look_rstat(savename,view="2022", write=True):
    db.cache_clear()
    if "opt" not in savename:
        save_name = "opt_"+savename
    else:
        save_name = savename
    nav, navsuffix = 1e8, "1e8"
    pd.set_option("display.width", 4000)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", None)
    success = 0
    if db.find(key=f"{save_name}_{navsuffix}_rstat"):
        print(savename)
        rstat = pd.DataFrame([db.read(date, f"{save_name}_{navsuffix}_rstat").rename(date) for date, _ in db.find(key=f"{save_name}_{navsuffix}_rstat") if date < view+"2"])
        perf = ms.annual_performance(rstat)
        if(write):
            rstatroot = f"/data/futures/lmdb/tcf/60s/features/yfcui1/opt/pred/{savename}/rstat.txt"
            with open(rstatroot,"w") as f:
                f.write(perf.to_string())
        print("rstat:")
        print(perf)
        success += 1
    
    if db.find(key=f"{save_name}_{navsuffix}_vcaprstat"):
        rstat = pd.DataFrame([db.read(date, f"{save_name}_{navsuffix}_vcaprstat").rename(date) for date, _ in db.find(key=f"{save_name}_{navsuffix}_vcaprstat") if date < view+"2"])
        vcapperf = ms.annual_performance(rstat)
        if(write):
            rstatroot = f"/data/futures/lmdb/tcf/60s/features/yfcui1/opt/pred/{savename}/vcaprstat.txt"
            with open(rstatroot,"w") as f:
                f.write(perf.to_string())
        print("vcaprstat:")
        print(vcapperf)
        success += 1
    if (success >= 2):
        return perf
    else:
        return None


# 获取每个因子在截面上的zscore
def get_zscore_xdata(alpha):
    alphaname, begin, end = alpha
    key = alphaname + "_decay"
    dts = [i[0] for i in db.find(key=key) if (i[0] >= begin) and (i[0] <= end)]
    allnames = set()
    vall = db.bulk_read(dts, key, read_only=True)
    univall = db.bulk_read(dts, "tm40c_universe", read_only=True)
    v = []
    for V, univ in zip(vall, univall):
        V = V[univ.index].T
        V = (V - V.mean()) / (V.std() + 1e-12)
        v += list(np.reshape(V.values, -1))
    return pd.Series(np.array(v)).rename(alphaname + "_zscore")

def all_zscore_xdata(alphaslist, begin, end):
    X = []
    alphas = zip(alphaslist, [begin] * len(alphaslist), [end] * len(alphaslist))
    with Pool(min(50, len(alphaslist))) as pool:
        for alpha, v in zip(alphaslist, tqdm(pool.imap(get_zscore_xdata, alphas), total=len(alphaslist))):
            X.append(v)
    return pd.concat(X, axis=1)

def build_future_name_index(name_index_dict = None, begin="20140101", end="20221229"):
    name_dict = dict()
    if(name_index_dict is not None):
        name_dict = name_index_dict.copy()
    key = "tm40c_return_oneday_close_ov"
    dts = [i[0] for i in db.find(key=key) if (i[0] >= begin) and (i[0] <= end)]
    allnames = set()
    vall = db.bulk_read(dts, key, read_only=True)
    univall = db.bulk_read(dts, "tm40c_universe", read_only=True)
    for V, univ in zip(vall, univall):
        names = set([x[:-4] for x in univ.index])
        allnames.update(names)
    allnames = sorted(list(allnames))
    for name in allnames:
        if(name not in name_dict):
            name_dict[name] = len(name_dict)
    return name_dict

def get_future_name_feature(begin="20140101", end="20221229", name_index_dict = None):
    name_dict = build_future_name_index(name_index_dict, begin, end)
    key = "tm40c_return_oneday_close_ov"
    dts = [i[0] for i in db.find(key=key) if (i[0] >= begin) and (i[0] <= end)]
    vall = db.bulk_read(dts, key, read_only=True)
    univall = db.bulk_read(dts, "tm40c_universe", read_only=True)
    future_names_feature = []
    for V, univ in zip(vall, univall):
        names = np.array([name_dict[x[:-4]] for x in univ.index]).repeat(V.shape[0])
        future_names_feature.append(names)
    future_names_feature = pd.Series(np.concatenate(future_names_feature)).rename("feature_future_name")
    return future_names_feature, name_dict

def legalize(modelname):
    root = f"{db.LMDB_ROOT}/tcf/60s/opt"
    model_path = modelname.split("_")
    model_path = os.path.join(root, *model_path)
    folder = os.path.join(model_path, "1e8")
    filekeys = [f"opt_{modelname}_{x[:-4]}" for x in os.listdir(model_path) if x.endswith("-mdb")]
    filekeys.extend([f"opt_{modelname}_1e8_{x[:-4]}" for x in os.listdir(folder) if x.endswith("-mdb")])
    locks = [os.path.join(model_path, x) for x in os.listdir(model_path) if x.endswith("-mdb-lock")]
    locks.extend([os.path.join(folder, x) for x in os.listdir(folder) if x.endswith("-mdb-lock")])
    for lockpath in locks:
        os.remove(lockpath)
    for mykey in filekeys:
        dts = [x[0] for x in db.find(key=mykey) if x[0] > "20211231"]
        db.delete_dates(dts, mykey)
        db.cache_clear()
    
    look_rstat(modelname, write=False)

def copy_rename(originname, newname):
    root = f"{db.LMDB_ROOT}/tcf/60s/opt"
    origin_path = originname.split("_")
    origin_path = os.path.join(root, *origin_path)
    if(not os.path.exists(origin_path)):
        raise ValueError(f"{origin_path} not exists")
    new_path = newname.split("_")
    new_path = os.path.join(root, *new_path)
    if(not os.path.exists(new_path)):
        os.makedirs(new_path)
    for filename in os.listdir(origin_path):
        if(filename.endswith("-mdb") or filename.endswith("-mdb-lock")):
            shutil.copy(os.path.join(origin_path, filename), os.path.join(new_path, filename))
        if(filename == "1e8"):
            origin_folder = os.path.join(origin_path, filename)
            new_folder = os.path.join(new_path, filename)
            if(not os.path.exists(new_folder)):
                os.makedirs(new_folder)
            for foldername in os.listdir(origin_folder):
                if(foldername.endswith("-mdb") or foldername.endswith("-mdb-lock")):
                    shutil.copy(os.path.join(origin_folder, foldername), os.path.join(new_folder, foldername))

