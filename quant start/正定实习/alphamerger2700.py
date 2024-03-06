import json
import logging
import os
import sys
import numpy as np
import pandas as pd
from futures import common
from futures_model import db
from cal_ic import reshape_data, reshape_data_acc, cal_ic, cal_ic_raw, reshape_data_acc_cut
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
import xgboost as xgb
import threading
import multiprocessing
from multiprocessing import Pool

from tqdm import tqdm

from matplotlib import pyplot as pyplot
import seaborn as sns

# 创建一个logger
logger = logging.getLogger('logger2700')
logger.setLevel(logging.DEBUG)
# 创建一个handler，用于写入日志文件
logger_path = '/home/yfcui1/futures-model/machine_learning/logger_merger2700.log'
fh = logging.FileHandler(logger_path)
fh.setLevel(logging.DEBUG)
# 创建一个handler，用于输出到控制台
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
# 定义handler的输出格式
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
# 给logger添加handler
logger.addHandler(fh)
logger.addHandler(ch)

beginend = {}
for year in range(2014, 2023):
    year = str(year)
    beginend[year] = [[i for i in common.find_dates() if i.startswith(year)][0],
                      [i for i in common.find_dates() if i.startswith(year)][-1]]
beginend["2022"][1] = "20221229"

def gen_xdata(alpha):
    alpha, begin, end = alpha
    xtrain = reshape_data_acc_cut(alpha + "_decay", begin, end)
    return pd.Series(xtrain).rename(alpha)

def all_xdata(alphaslist, begin, end, parallel = True):
    X = []
    alphas = zip(alphaslist, [begin] * len(alphaslist), [end] * len(alphaslist))
    if(parallel):
        with Pool(min(50, len(alphaslist))) as pool:
            for alpha, v in zip(alphaslist, pool.map(gen_xdata, alphas)):
                X.append(v)
    else:
        for alpha, v in zip(alphaslist, map(gen_xdata, alphas)):
            X.append(v)
    return pd.concat(X, axis=1)

def alphalist_split(alphaslist):
    tmp_list = sorted(alphaslist)
    alpha_groups = dict()
    for alphaname in tmp_list:
        code = alphaname.split("_")[4]
        if(code not in alpha_groups):
            alpha_groups[code] = []
        alpha_groups[code].append(alphaname)

    return alpha_groups

def get_ic(y_pred, y_true):
    y_pre = y_pred.values
    y_tru = y_true.values
    return np.corrcoef(y_pre, y_tru)[0, 1]

def get_r2(y_pred, y_true):
    y_pre = y_pred.values
    y_tru = y_true.values
    return 1 - np.sum((y_pre - y_tru) ** 2) / np.sum((y_tru - y_tru.mean()) ** 2)


# 根据log断点续传
def continue_by_log():
    pass

def start_by_log():
    pass

def merge_one_alphagroup(year, alphacode, alphalist, label_train, label_test, save_path, continue_log_path = None):
    universe = "tm40c"
    train_years = np.arange(year-5, year)
    start = beginend[str(train_years[0])][0]
    end = beginend[str(train_years[-1])][1]

    f = reshape_data_acc(f"{universe}_daynight_frame", start, end)
    data_train = all_xdata(alphalist, start, end, parallel=False)[f>0]
    # print(f"data_train.shape={data_train.shape}, label_train.shape={label_train.shape}")
    model = Ridge(alpha=1.0)
    model.fit(data_train, label_train)
    f = reshape_data_acc(f"{universe}_daynight_frame", beginend[str(year)][0], beginend[str(year)][1])
    data_test = all_xdata(alphalist, beginend[str(year)][0], beginend[str(year)][1], parallel=False)[f>0]
    alphaname = "_".join(alphalist[0].split('_')[:5]) + "_ridge1"
    pred_train = pd.Series(model.predict(data_train)).rename(alphaname)
    pred_test = pd.Series(model.predict(data_test)).rename(alphaname)
    train_ic = get_ic(pred_train, label_train)
    train_r2 = get_r2(pred_train, label_train)
    test_ic = get_ic(pred_test, label_test)
    test_r2 = get_r2(pred_test, label_test)
    if (not os.path.exists(f"{save_path}/{year}/")):
        os.makedirs(f"{save_path}/{year}/")
    pred_train.to_pickle(f"{save_path}/{year}/{alphacode}_{np.min(train_years)}-{np.max(train_years)}.pickle")
    pred_test.to_pickle(f"{save_path}/{year}/{alphacode}_{year}.pickle")
    # logger.info(f"{year}-{alphacode}, train_ic={train_ic}, test_ic={test_ic}, train_r2={train_r2}, test_r2{test_r2}, merging alphas...Done!")
    logger.info("%d-%s, train_ic=%.5f, test_ic=%.5f, train_r2=%.5f, test_r2=%.5f, merging alphas...Done!" % (year, alphacode, train_ic, test_ic, train_r2, test_r2))

    return

def merge_alpha_per_year(year, alphagroups, save_folder, process_queue, continue_log_path = None):
    train_years = np.arange(year-5, year)
    start = beginend[str(train_years[0])][0]
    end = beginend[str(train_years[-1])][1]

    f = reshape_data_acc(f"{universe}_daynight_frame", start, end)
    label_train = pd.Series(reshape_data_acc("tm40c_return_oneday_close_ov", start, end)[f>0]).rename("tm40c_return_oneday_close_ov")
    f = reshape_data_acc(f"{universe}_daynight_frame", beginend[str(year)][0], beginend[str(year)][1])
    label_test = pd.Series(reshape_data_acc("tm40c_return_oneday_close_ov", beginend[str(year)][0], beginend[str(year)][1])[f>0]).rename("tm40c_return_oneday_close_ov")
    logger.info(f"{year}, labels for %d...Done!")

    logger.info(f"{year}, merging alphas...")
    # with Pool(min(15, len(alphagroups))) as pool:
    #     for alphacode, alphalist in alphagroups.items():
    #         pool.starmap(merge_one_alphagroup, [(year, alphacode, alphalist, label_train, label_test, save_folder, continue_log_path)])
    max_workers = min(80, len(alphagroups))
    num_workers = 0
    joblist = []
    for i, alphacode in enumerate(alphagroups):
        alphalist = alphagroups[alphacode]
        if(num_workers == max_workers):
            for job in joblist:
                job.start()
            for job in joblist:
                job.join()
            num_workers = 0
            joblist = []
        elif(i == len(alphagroups) - 1):
            num_workers += 1
            job = multiprocessing.Process(target=merge_one_alphagroup, args=(year, alphacode, alphalist, label_train, label_test, save_folder, continue_log_path))
            joblist.append(job)
            for job in joblist:
                job.start()
            for job in joblist:
                job.join()
            num_workers = 0
            joblist = []
        else:
            num_workers += 1  
            job = multiprocessing.Process(target=merge_one_alphagroup, args=(year, alphacode, alphalist, label_train, label_test, save_folder, continue_log_path))
            joblist.append(job)
    logger.info(f"{year}, merging alphas...Done!")

    return

# 流程
# 先把alphslist按照名字排序，膜三分成3份
# 接着对于每一份，做逐年rolling筛因子。对于每一年，先把所有因子的IC算出来，然后按照IC排序，从大到小加入因子，用岭回归判定是否加入
def rolling_merge_alphas(alphaslist, save_folder = "/data/futures/lmdb/tcf/60s/features/yfcui1/tm40c/data2700/merger_2700", continue_log_path = None):
    alphagroups = alphalist_split(alphaslist)
    
    process_queue = None
    years = np.arange(2019, 2023)
    # with Pool(min(15, len(years))) as pool:
    #     pool.starmap(merge_alpha_per_year, [(year, alphagroups, save_folder, process_queue, continue_log_path) for year in years])
    for year in years:
        merge_alpha_per_year(year, alphagroups, save_folder, process_queue, continue_log_path)
    
    for year in years:
        logger.info(f"{year}, cancatenating alphas...")
        train_years = np.arange(year-5, year)
        if (not os.path.exists(f"{save_folder}/{year}/")):
            os.makedirs(f"{save_folder}/{year}/")
        merged_alphas_train = sorted([x for x in os.listdir(f"{save_folder}/{year}/") if x.endswith(f"{np.min(train_years)}-{np.max(train_years)}.pickle")])
        concat_data_train = pd.concat([pd.read_pickle(f"{save_folder}/{year}/{x}") for x in tqdm(merged_alphas_train)], axis=1)
        concat_data_train.to_pickle(f"{save_folder}/X_{np.min(train_years)}-{np.max(train_years)}.pickle")
        del concat_data_train
        merged_alphas_test = sorted([x for x in os.listdir(f"{save_folder}/{year}/") if x.endswith(f"{year}.pickle")])
        concat_data_test = pd.concat([pd.read_pickle(f"{save_folder}/{year}/{x}") for x in tqdm(merged_alphas_test)], axis=1)
        concat_data_test.to_pickle(f"{save_folder}/X_{year}.pickle")
        del concat_data_test
        logger.info(f"{year}, cancatenating alphas...Done!")


if __name__ == "__main__":
    universe = "tm40c"
    root = f"{db.LMDB_ROOT}/tcf/60s/features{universe}"
    config_path = "/home/yfcui1/futures-model/machine_learning/featuresall2700.json"
    with open(config_path) as fin:
        config = json.load(fin)

    alphaslist = list(config["alphas"].keys())
    rolling_merge_alphas(alphaslist, save_folder = "/data/futures/lmdb/tcf/60s/features/yfcui1/tm40c/data2700/merger_2700", continue_log_path = None)