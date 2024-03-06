# %%
import json
import logging
import os
import sys
from tqdm import tqdm
import numpy as np
import pandas as pd
from futures import common
from futures_model import db
from cal_ic import reshape_data, reshape_data_acc, cal_ic, get_ic, cal_ic_raw, reshape_data_acc_cut
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import xgboost as xgb
import threading
import multiprocessing
from multiprocessing import Pool
from sklearn.linear_model import Ridge, LinearRegression, Lasso

import seaborn as sns
import matplotlib.pyplot as plt

from cyfutils import group_plot, get_preds, merge_preds, print_pred_score, save_pred_model, get_xgbmodel_config, cal_pred_tvr, get_alpha_stat, get_zscore_xdata, all_zscore_xdata, build_future_name_index, get_future_name_feature
from bcorr import cmp_bcorr

# %%
universe = "tm40c"
root = f"{db.LMDB_ROOT}/tcf/60s/features{universe}"
root

# %%
config_path = "/home/yfcui1/futures-model/machine_learning/featuresall2700.json"
with open(config_path) as fin:
    config = json.load(fin)
alphaslist = list(config["alphas"].keys())

# %%
beginend = {}
for year in range(2014, 2023):
    year = str(year)
    beginend[year] = [[i for i in common.find_dates() if i.startswith(year)][0],
                      [i for i in common.find_dates() if i.startswith(year)][-1]]
beginend["2022"][1] = "20221229"

# %%
alphalist = dict()
alphadf = pd.read_csv("/home/yfcui1/futures-model/machine_learning/tmc_selected_alphas2700_split2.csv", index_col=0)
# alphadf = pd.read_csv("/home/yfcui1/futures-model/machine_learning/tmc_selected_alphas2700_3splits_merge.csv", index_col=0)
for year in alphadf.columns:
    alphalist[year] = alphadf[year].dropna().tolist()
del alphadf
common_alphalist = [set(alphalist[year]) for year in alphalist]
common_alphalist = list(set.union(*common_alphalist))

# %%
label = dict()
for year in beginend.keys():
    f = reshape_data_acc(f"{universe}_daynight_frame", beginend[year][0], beginend[year][1])
    label[year] = reshape_data_acc_cut("tm40c_return_oneday_close_ov", beginend[year][0], beginend[year][1])
    label[year] = label[year][f>0]
    label[year] = (pd.Series(label[year]).rename("tm40c_return_oneday_close_ov")).reset_index(drop=True)
label["train"] = pd.concat([label[str(year)] for year in np.arange(2014, 2019)], axis=0)
label["cv"] = label["2019"]
label["test"] = label["2020"]

# %%
# data = dict()
# for year in beginend.keys():
#     f = reshape_data_acc(f"{universe}_daynight_frame", beginend[year][0], beginend[year][1])
#     data[year] = all_xdata(common_alphalist, beginend[year][0], beginend[year][1])[f >0]
#     print(f"{year} data loaded.")
data = dict()
# for year in beginend.keys():
#     data[year] = pd.read_pickle(f"/data/futures/lmdb/tcf/60s/features/yfcui1/tm40c/data2700/split2/X_{year}.pickle")
#     print(f"{year} data loaded.")

from sklearn.manifold import MDS
sigtot = pd.read_csv("/data/futures/lmdb/tcf/60s/features/sig2021.csv", index_col=0)
diag = 1/np.sqrt(np.diagonal(sigtot.values)).reshape((-1,1))
corrmatrix = (diag @ diag.T)*sigtot
cos_dist = 1 - corrmatrix
mds = MDS(n_components=1, dissimilarity='precomputed', random_state=42)
coords = mds.fit_transform(cos_dist)
coords = pd.DataFrame(coords, index=corrmatrix.columns, columns=["x"])
mapping = {col: coords[col].to_dict() for col in coords.columns}

futures_name_dict = None
futures_name_dict_reverse = None
feature_futures_name = dict()
for year in beginend.keys():
    f = reshape_data_acc(f"{universe}_daynight_frame", beginend[year][0], beginend[year][1])
    feature_futures_name[year], futures_name_dict = get_future_name_feature(beginend[year][0], beginend[year][1], futures_name_dict)
    futures_name_dict_reverse = {v: k for k, v in futures_name_dict.items()}
    feature_futures_name[year] = feature_futures_name[year][f>0].reset_index(drop=True)
    feature_futures_name[year] = feature_futures_name[year].map(futures_name_dict_reverse)
    feature_futures_name[year] = pd.concat([(feature_futures_name[year].map(mapping[x])).rename(f"feature_futures_name_{x}") for x in mapping.keys()], axis=1)
    print(f"{year} feature_futures_name loaded, futures_name_dict updated.")

# %%
# data_zscore = dict()
# for year in beginend.keys():
#     f = reshape_data_acc(f"{universe}_daynight_frame", beginend[year][0], beginend[year][1])
#     data_zscore[year] = all_zscore_xdata(common_alphalist, beginend[year][0], beginend[year][1])[f >0]
#     print(f"{year} zscore_data loaded.")
# data_zscore = dict()
# for year in beginend.keys():
#     data_zscore[year] = pd.read_pickle(f"/data/futures/lmdb/tcf/60s/features/yfcui1/tm40c/data2700/split0/Xzscore_{year}.pickle")
#     print(f"{year} zscore_data loaded.")


# %%
def train_predict_1_year(models, predyear, data, label, alphanames, 
                         data_zscore = None, 
                         feature_futures_name = None, 
                         is_MC = True, 
                         is_expanding = False,
                         is_std = False,
                         process_queue = None):
    year_dict = {"2019": "cv", "2020": "test", "2021": "2021", "2022": "2022"}
    if(is_expanding):
        train_years = np.arange(2014, predyear)
    else:
        train_years = np.arange(predyear-5, predyear)
    train_suffix = f"{np.min(train_years)}-{np.max(train_years)}"
    test_suffix = f"{predyear}"
    print("read data for year %s start!" % predyear)
    # Xtrain = pd.concat([data[str(year)][alphanames] for year in train_years], axis=0)
    Xtrain = pd.read_pickle(f"/data/futures/lmdb/tcf/60s/features/yfcui1/tm40c/data2700/merger_2700/X_{train_suffix}.pickle").reset_index(drop=True)
    # print(f"loaded X_{train_suffix}.pickle")
    if(is_std):
        std = Xtrain.std()
        Xtrain = Xtrain / std
    if(data_zscore is not None):
        zscore_alphanames = [i+"_zscore" for i in alphanames]
        Xtrain_zscore = pd.concat([data_zscore[str(year)][zscore_alphanames] for year in train_years], axis=0)
        Xtrain = pd.concat([Xtrain, Xtrain_zscore], axis=1)
        del Xtrain_zscore
    ytrain = pd.concat([label[str(year)] for year in train_years], axis=0)

    if(is_MC):
        Xtrain = pd.concat([Xtrain, -Xtrain], axis=0).reset_index(drop=True)
        ytrain = pd.concat([ytrain, -ytrain], axis=0).reset_index(drop=True)
    mean = ytrain.mean()
    sigma3 = ytrain.std() * 3
    ytrain = ytrain.clip(mean-sigma3, mean+sigma3)
    if(feature_futures_name is not None):
        # futures_name_train = pd.concat([feature_futures_name[str(year)] for year in np.arange(2014, predyear)], axis=0)
        # category = futures_name_train.unique().tolist()
        futures_name_train = pd.concat([feature_futures_name[str(year)] for year in train_years], axis=0).reset_index(drop=True)
        if(is_MC):
            futures_name_train = pd.concat([futures_name_train, futures_name_train], axis=0).reset_index(drop=True)
        # futures_name_train = pd.Categorical(futures_name_train, categories=category)
        Xtrain = pd.concat([Xtrain, futures_name_train], axis=1) 
        print("add feature_future_name for year %s success!"%(predyear))
    print("read data for year %s success!" % predyear)

    print("train model for year %s start!" % predyear)
    joblist = []
    for model in models:
        job = threading.Thread(target=model.fit, args=(Xtrain, ytrain))
        joblist.append(job)
    for job in joblist:
        job.start()
    for job in joblist:
        job.join()
    print("train model for year %s success!" % predyear)

    print("predict model for year %s start!" % predyear)
    Xtest = dict()
    if(predyear == 2019):
        Xtest["train"] = Xtrain.iloc[:int(Xtrain.shape[0]/2)] if is_MC else Xtrain
    del Xtrain, ytrain    
    # Xtest[year_dict[str(predyear)]] = pd.concat([data[str(predyear)][alphanames]])
    Xtest[year_dict[str(predyear)]] = pd.read_pickle(f"/data/futures/lmdb/tcf/60s/features/yfcui1/tm40c/data2700/merger_2700/X_{test_suffix}.pickle").reset_index(drop=True)
    if(is_std):
        Xtest[year_dict[str(predyear)]] = Xtest[year_dict[str(predyear)]] / std
    if(data_zscore is not None):
        Xtest_zscore = data_zscore[str(predyear)][zscore_alphanames]
        Xtest[year_dict[str(predyear)]] = pd.concat([Xtest[year_dict[str(predyear)]], Xtest_zscore], axis=1)
        del Xtest_zscore
    if(feature_futures_name is not None):
        # futures_name_test = pd.Categorical(pd.concat([feature_futures_name[str(predyear)]]), categories=category)
        futures_name_test = feature_futures_name[str(predyear)]
        Xtest[year_dict[str(predyear)]] = pd.concat([Xtest[year_dict[str(predyear)]], futures_name_test], axis=1)
    print("read data for year %s success!" % predyear)
    if(predyear == 2019):
        myrange = ["train", year_dict[str(predyear)]]
    else:
        myrange = [year_dict[str(predyear)]]
    joblist = []
    pid_queue = multiprocessing.Queue()
    for model in models:
        job = threading.Thread(target=get_preds, args=(model, Xtest, myrange, pid_queue))
        joblist.append(job)
    for job in joblist:
        job.start()
    for job in joblist:
        job.join()
    model_preds_list = [pid_queue.get() for job in joblist]
    model_preds_list = sorted(model_preds_list, key=lambda x: x[0].save_name)
    # model_preds = get_preds(model, Xtest, range = myrange, process_queue = pid_queue)
    print("predict model for year %s success!" %(predyear))
    for model_preds in model_preds_list:
        save_pred_model(model_preds[1], model_preds[0].save_name, model = model_preds[0], range = myrange)
    if(process_queue is not None):
        process_queue.put(model_preds_list)
    else:
        return model_preds_list

# %%
def train_predict_years(data, label, alphalist, 
                        savename = "cyftmc2700_split1_gbtree00_rollic_zscore",
                        data_zscore = None, 
                        feature_futures_name = None, 
                        is_MC = True, 
                        is_std = False,
                        is_expanding = False, 
                        parrallel = False):
    predyears = np.arange(2019, 2023)
    year_dict = {"2019": "cv", "2020": "test", "2021": "2021", "2022": "2022"}
    save_range = []
    save_range.extend([year_dict[str(year)] for year in predyears])
    look_range = ["2022"]
    pid_queue = None
    joblist = None
    if(parrallel):
        pid_queue = multiprocessing.Queue()
        joblist = []
    model_preds = dict()
    for year in predyears:
        used_features = alphalist[str(year)]
        # random_seeds = [0, 1, 2, 3, 4, 5, 6, 7, 42, 1024]
        params = [(300, 0.0067, 4), (300, 0.0067, 3)]
        models = []
        for para in params:
            model = XGBRegressor(
                            booster="gbtree",
                            n_estimators=para[0],
                            max_depth=para[2],
                            learning_rate=para[1],
                            min_child_weight=1,
                            subsample=1,
                            colsample_bytree=1,
                            gamma=0,
                            reg_alpha=0,
                            reg_lambda=0,
                            verbosity=3,
                            #nthread=nthread,
                            eval_metric="rmse",
                            objective="reg:squarederror",
                            seed=10,
                            enable_categorical=True,
                            )
            # model = Ridge(alpha = 1)
            model.save_name = (savename + f"n{para[0]}lr{para[1]}d{para[2]}").replace(".", "")
            models.append(model)
        if(not parrallel):
            tmp_model_preds_list = train_predict_1_year(models, year, data, label, used_features, data_zscore, feature_futures_name, is_MC, is_std, is_expanding)
            if(len(model_preds) == 0):
                model_preds = {x[0].save_name: x[1] for x in tmp_model_preds_list}
            else:
                for tmp_model_preds in tmp_model_preds_list:
                    model_preds[tmp_model_preds[0].save_name].update(tmp_model_preds[1])
        else:
            job = threading.Thread(target=train_predict_1_year, args=(models, year, data, label, used_features, data_zscore, feature_futures_name, is_MC, is_std, is_expanding, pid_queue))
            joblist.append(job)
    if(parrallel):
        max_workers = parrallel
        num_workers = 0
        tmp_joblist = []
        for ii, job in enumerate(joblist):
            tmp_joblist.append(job)
            num_workers += 1   
            if(num_workers == max_workers):
                for job in tmp_joblist:
                    job.start()
                for job in tmp_joblist:
                    job.join()
                num_workers = 0
                tmp_joblist = []
            elif(ii == len(joblist) - 1):
                for job in tmp_joblist:
                    job.start()
                for job in tmp_joblist:
                    job.join()
                num_workers = 0
                tmp_joblist = []

        tmp_model_preds_list = [pid_queue.get() for job in joblist]
        for tmp_model_preds in tmp_model_preds_list:
            if(len(model_preds) == 0):
                model_preds = {x[0].save_name: x[1] for x in tmp_model_preds}
            else:
                for tmp_preds in tmp_model_preds:
                    model_preds[tmp_preds[0].save_name].update(tmp_preds[1])
    for save_names in model_preds.keys():
        model_results = print_pred_score(model_preds[save_names], 
                        label, 
                        range = look_range, 
                        #  scores = ["RMSE", "IC", "Rank_IC", "Binary_acc", "Binary_true_positive", "Binary_true_negative", "Binary_F1", "R2"], 
                        group_show = False)
    
        save_pred_model(model_preds[save_names], save_names, range = save_range)
        model_config = get_xgbmodel_config(model_preds[save_names], model_results, save_names, alphalist=alphalist, save = True)
    return model_preds

# %%
model1_preds = train_predict_years(data, 
                                                                  label, 
                                                                  alphalist, 
                                                                  savename = "cyftmc2700_merge530_gbtree00_rollic",
                                                                  data_zscore = None, # data_zscore = data_zscore,
                                                                  feature_futures_name = None, # feature_futures_name = feature_futures_name,
                                                                  is_MC = False, 
                                                                  is_std = False,
                                                                  is_expanding = False, 
                                                                  parrallel = 2)

# %%
keys = list(model1_preds.keys())

os.system("python3.8 /home/yfcui1/futures-model/machine_learning/vcapshow.py -s %s"%(",".join(keys)))