import json
import logging
import os
import sys
import numpy as np
import pandas as pd
from futures import common
from futures_model import db
from cal_ic import reshape_data, reshape_data_acc, cal_ic, get_ic, cal_ic_raw, reshape_data_acc_cut
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
logger_path = '/home/yfcui1/futures-model/machine_learning/logger2700.log'
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

def all_xdata(alphaslist, begin, end):
    X = []
    alphas = zip(alphaslist, [begin] * len(alphaslist), [end] * len(alphaslist))
    with Pool(min(50, len(alphaslist))) as pool:
        for alpha, v in zip(alphaslist, pool.map(gen_xdata, alphas)):
            X.append(v)
    return pd.concat(X, axis=1)

def alphalist_split(alphaslist, num_split = 3):
    tmp_list = sorted(alphaslist)
    alphadict= dict()
    for i in range(num_split):
        alphadict["split%d" % i] = []
    for i, item in enumerate(tmp_list):
        alphadict["split%d" % (i%num_split)].append(item)
    return alphadict

def get_bulk_name(alphalist, maxnum = 50):
    num_bulks = len(alphalist)//maxnum if len(alphalist)%maxnum == 0 else len(alphalist)//maxnum + 1
    bulks = []
    for i in range(num_bulks):
        bulks.append(alphalist[i*maxnum:min((i+1)*maxnum, len(alphalist))])
    return bulks

def get_IC(x, y, process_queue):
    tp = (x.name, np.corrcoef(x.values, y.values)[0, 1])
    process_queue.put(tp)
    return tp

def parrallel_cal_ic(data, label):
    process_queue = multiprocessing.Queue()
    joblist = []

    for alpha in data.columns:
        joblist.append(threading.Thread(target = get_IC, args = (data[alpha], label, process_queue)))
    for job in joblist:
        job.start()
    for job in joblist:
        job.join()
    ic = [process_queue.get() for job in joblist]
    ic = pd.Series([x[1] for x in ic], index = [x[0] for x in ic])
    return ic

def get_bulk_alpha_IC(alphalist, begin, end, label, maxnum = 50, issort = True):
    bulks = get_bulk_name(alphalist, maxnum)

    ic = dict()
    f = reshape_data_acc(f"{universe}_daynight_frame", begin, end)
    for alphabulk in tqdm(bulks):
        data = all_xdata(alphabulk, begin, end)
        data = data[f>0]
        # ics = data.corrwith(label)
        ics = parrallel_cal_ic(data, label)
        for alpha in alphabulk:
            ic[alpha] = ics[alpha]
    if(issort):
        ic = sorted(ic.items(), key = lambda x: x[1], reverse = True)
    return [x[0] for x in ic]

# 根据log断点续传
def continue_by_log(item, year, sorted_alphalist, log_path):
    if(not os.path.exists(log_path)):
        return [], sorted_alphalist, 0, 0, 0
    with open(log_path) as fin:
        lines = fin.readlines()
    mylines = []
    for line in lines:
        tmp_line = [x.strip() for x in line.split("-")]
        if(len(tmp_line) < 7):
            continue
        if(tmp_line[5] == item and tmp_line[6].startswith(str(year))):
            tmp = tmp_line[6].split(",")[1].strip()
            if(tmp.startswith("add") or tmp.startswith("skip")):
                mylines.append(tmp_line[6])
    del lines
    addlist = []
    droplist = []
    usedlist = []
    keeped_alphalist = []
    remain_sorted_alphas = []
    best_score = 0
    order = 0
    num_alphas = 0

    if(len(mylines) > 0):
        lastline = [x.strip() for x in mylines[-1].split(",")]
        for tmp in lastline:
            if(tmp.startswith("best_score")):
                best_score = float(tmp.split("=")[1].strip())
            elif(tmp.startswith("order")):
                order = int(tmp.split("=")[1].strip())
            elif(tmp.startswith("num_alphas")):
                num_alphas = int(tmp.split("=")[1].strip())

        for line in mylines:
            tmp_line = [x.strip() for x in line.split(",")]
            for sentence in tmp_line:
                if(sentence.startswith("add")):
                    tmp = sentence.split(" ")[1].strip()
                    addlist.append(tmp)
                    usedlist.append(tmp)
                elif(sentence.startswith("drop")):
                    tmp_sentences = [x.strip() for x in sentence.split(" ") if x.strip() != "drop"]
                    droplist.extend(tmp_sentences)
                    usedlist.extend(tmp_sentences)
                elif(sentence.startswith("skip")):
                    tmp = sentence.split(" ")[1].strip()
                    usedlist.append(tmp)
        usedlist = set(usedlist)
        remain_sorted_alphas = [x for x in sorted_alphalist if x not in usedlist]
        keeped_alphalist = list(set(addlist).difference(set(droplist)))

    return keeped_alphalist, remain_sorted_alphas, best_score, order, num_alphas

def start_by_log(item, year, log_path):
    if(log_path is None):
        return False
    if(not os.path.exists(log_path)):
        return False
    with open(log_path) as fin:
        lines = fin.readlines()
    mylines = []
    for line in lines:
        tmp_line = [x.strip() for x in line.split("-")]
        if(len(tmp_line) < 7):
            continue
        if(tmp_line[5] == item and tmp_line[6].startswith(str(year))):
            tmp = tmp_line[6].split(",")[1].strip()
            if(tmp == "selecting alphas...Done!"):
                return True
    return False

def select_alpha_per_year(alphalist, item, year, process_queue, continue_log_path = None):
    train_years = np.arange(year-5, year)
    start = beginend[str(train_years[0])][0]
    end = beginend[str(train_years[-1])][1]

    f = reshape_data_acc(f"{universe}_daynight_frame", start, end)
    label = pd.Series(reshape_data_acc("tm40c_return_oneday_close_ov", start, end)[f>0]).rename("tm40c_return_oneday_close_ov")
    logger.info("%s, labels for %d...Done!" %(item, year))

    logger.info("%s-%s, sorting alphas..."%(item, year))
    sorted_alphas = get_bulk_alpha_IC(alphalist, start, end, label, maxnum = 50, issort = True)
    logger.info("%s-%s, sorting alphas...Done!"%(item, year))

    keeped_data = None
    selected_alphas = []
    best_score = 0
    total_num = 0
    if(continue_log_path is not None):
        logger.info("%s-%s, continue by log..."%(item, year))
        keeped_alphas, remain_sorted_alphas, best_score, total_num, num_alphas = continue_by_log(item, year, sorted_alphas, log_path=continue_log_path)
        if(len(keeped_alphas) > 0):
            selected_alphas = keeped_alphas
            sorted_alphas = remain_sorted_alphas
            keeped_data = all_xdata(selected_alphas, start, end)
            keeped_data = keeped_data[f>0]
            keeped_data = keeped_data/(keeped_data.std())
            logger.info("%s-%s, continue by log...Done!"%(item, year))
        else:
            logger.info("%s-%s, continue by log...Failed!"%(item, year))

    alpha_bulks = get_bulk_name(sorted_alphas, maxnum = 50)
    logger.info("%s-%s, selecting alphas..."%(item, year))
    for i, alphabulk in enumerate(alpha_bulks):
        data = all_xdata(alphabulk, start, end)
        data = data[f>0]
        for alphaname in alphabulk:
            total_num += 1
            if(len(selected_alphas) == 0):
                selected_alphas.append(alphaname)
                keeped_data = pd.DataFrame(data[alphaname]/(data[alphaname].std()))
                model = Ridge(alpha = 1)
                model.fit(keeped_data, label)
                pred = model.predict(keeped_data)
                score = np.corrcoef(pred, label.values)[0, 1]
                best_score = score
                logger.info("%s-%s, add %s, score=%f, best_score=%f, order=%d, num_alphas=%d" %(item, year, alphaname, score, best_score, total_num, len(selected_alphas)))
            else:
                # corr = keeped_data.corrwith(data[alphaname])
                corr = parrallel_cal_ic(keeped_data, data[alphaname])
                if(corr.max() <= 0.7):
                    selected_alphas.append(alphaname)
                    keeped_data[alphaname] = data[alphaname]/(data[alphaname].std())
                    model = Ridge(alpha = 1)
                    model.fit(keeped_data, label)
                    pred = model.predict(keeped_data)
                    score = np.corrcoef(pred, label.values)[0, 1]
                    best_score = score
                    logger.info("%s-%s, add %s, score=%f, best_score=%f, order=%d, num_alphas=%d" %(item, year, alphaname, score, best_score, total_num, len(selected_alphas)))
                else:
                    conflicts = corr[corr>0.7].index.tolist()
                    the_rest = [x for x in selected_alphas if x not in conflicts]
                    conflict_data = keeped_data[conflicts]
                    keeped_data.drop(conflicts, axis=1, inplace=True)
                    keeped_data[alphaname] = data[alphaname]/(data[alphaname].std())
                    model = Ridge(alpha = 1)
                    model.fit(keeped_data, label)
                    pred = model.predict(keeped_data)
                    score = np.corrcoef(pred, label.values)[0, 1]
                    if(score >= best_score):
                        selected_alphas = the_rest
                        selected_alphas.append(alphaname)
                        best_score = score
                        logger.info("%s-%s, add %s, drop %s, score=%f, best_score=%f, order=%d, num_alphas=%d" %(item, year, alphaname, " ".join(conflicts), score, best_score, total_num, len(selected_alphas)))
                    else:
                        keeped_data.drop(alphaname, axis=1, inplace=True)
                        keeped_data = pd.concat([keeped_data, conflict_data], axis=1)
                        logger.info("%s-%s, skip %s, score=%f, best_score=%f, order=%d, num_alphas=%d" %(item, year, alphaname, score, best_score, total_num, len(selected_alphas)))
    logger.info("%s-%s, selecting alphas...Done!"%(item, year))
    selected_alphas = pd.Series(selected_alphas).rename(str(year))
    process_queue.put(selected_alphas)
    return selected_alphas

# 流程
# 先把alphslist按照名字排序，膜三分成3份
# 接着对于每一份，做逐年rolling筛因子。对于每一年，先把所有因子的IC算出来，然后按照IC排序，从大到小加入因子，用岭回归判定是否加入
def rolling_select_alphas(alphaslist, save_folder = "/home/yfcui1/futures-model/machine_learning", continue_log_path = None):
    alphadict = alphalist_split(alphaslist, 3)
    
    for bulkname, alphalist in alphadict.items():
        logger.info("selecting alphas for %s..." %(bulkname))
        selected_alphas = None
        process_queue = multiprocessing.Queue()
        job_list = []
        for year in range(2019, 2023):
            if(not start_by_log(bulkname, year, continue_log_path)):
                job_list.append(multiprocessing.Process(target=select_alpha_per_year, args=(alphalist, bulkname, year, process_queue, continue_log_path)))
        for job in job_list:
            job.start()
        for job in job_list:
            job.join()
        if(len(job_list) > 0):
            selected_alphas = sorted([process_queue.get() for job in job_list], key=lambda x: x.name)
            selected_alphas = pd.concat(selected_alphas, axis=1)
            selected_alphas.to_csv(f"{save_folder}/tmc_selected_alphas2700_{bulkname}.csv")
        logger.info("%s, selecting alphas for %d...Done!" %(bulkname, year))


if __name__ == "__main__":
    universe = "tm40c"
    root = f"{db.LMDB_ROOT}/tcf/60s/features{universe}"
    config_path = "/home/yfcui1/futures-model/machine_learning/featuresall2700.json"
    with open(config_path) as fin:
        config = json.load(fin)

    alphaslist = list(config["alphas"].keys())
    rolling_select_alphas(alphaslist, save_folder = "/home/yfcui1/futures-model/machine_learning", continue_log_path = logger_path)