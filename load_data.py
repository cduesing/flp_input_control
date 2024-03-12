import numpy as np
from numpy import dot
from numpy.linalg import norm
import pandas as pd
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from collections import Counter
from sklearn.cluster import KMeans
import sys
import statistics
import torch
import torchvision
import torchvision.transforms as transforms
import random
import skews


# load the customer churn dataset
def load_churn_dataset(config, column="2_digi_code"):

    config["num_features"] = 37
    config["num_classes"] = 2

    if config["model_name"] == "auto":
        config["model_name"] = "ann"

    df = pd.read_csv("./data/churn/churn.csv")
    y = list(df["is_churn"].to_numpy().astype(int))

    if column == "2_digi_code":
        zips = df["2_digi_code"].to_numpy()
        states = df["state"].to_numpy()
        zips_to_states = {}
        for i, code in enumerate(zips):
            if code not in zips_to_states:
              zips_to_states[code] = states[i]
        config["zips_to_states"] = zips_to_states

    clients = list(df[column].to_numpy())
    df = df.drop(["is_churn", "2_digi_code", "state", "recency_by_day"], axis=1)
    X = list(df.to_numpy().astype(float))

    code_to_id_dict = {x:i for i,x in enumerate(list(set(clients)))}
    id_to_code_dict = {v: k for k, v in code_to_id_dict.items()}

    clients_feature_dict = {i:[] for i in range(len(list(set(clients))))}
    clients_label_dict = {i:[] for i in range(len(list(set(clients))))}
    sample_to_client_assignment = {i:[] for i in range(len(list(set(clients))))}

    for i, x in enumerate(X):
        client_id = code_to_id_dict[clients[i]]
        clients_feature_dict[client_id].append(list(x))
        clients_label_dict[client_id].append(y[i])
        sample_to_client_assignment[client_id].append(i)

    config["code_dict"] = code_to_id_dict
    config["reverse_code_dict"] = id_to_code_dict

    config["clients_feature_dict"] = clients_feature_dict
    config["clients_label_dict"] = clients_label_dict

    config["sample_to_client_assignment"] = sample_to_client_assignment
    config["num_clients"] = len(list(set(clients)))
    config["client_group"] = column

    return config

# spit into train and test data according to region configuration
def split_train_test(config, region=None, frac=0.2, exclude=None):

    region_d = {
        "AC":"North",
        "AL":"Northeast",
        "AP":"North",
        "AM":"North",
        "BA":"Northeast",
        "CE":"Northeast",
        "DF":"Center West",
        "ES":"Southeast",
        "GO":"Center West",
        "MA":"Northeast",
        "MT":"Center West",
        "MS":"Center West",
        "MG":"Southeast",
        "PA":"North",
        "PB":"Northeast",
        "PR":"South",
        "PE":"Northeast",
        "PI":"Northeast",
        "RJ":"Southeast",
        "RN":"Northeast",
        "RS":"South",
        "RO":"North",
        "RR":"North",
        "SC":"South",
        "SP":"Southeast",
        "SE":"Northeast",
        "TO":"North",
    }
        
    if region == None:
        l = list(config["clients_feature_dict"].keys())
        random.shuffle(l)
        test_ids = l[:int(len(l)*frac)]
        train_ids = l[int(len(l)*frac):]

        train_feature_dict = {key: config["clients_feature_dict"][key] for key in train_ids}
        train_label_dict = {key: config["clients_label_dict"][key] for key in train_ids}

        test_feature_dict = {key: config["clients_feature_dict"][key] for key in test_ids}
        test_label_dict = {key: config["clients_label_dict"][key] for key in test_ids}
        

    elif isinstance(region, float):
        l = list(config["clients_feature_dict"].keys())
        if "zips_to_states" in config:
            states = [config["zips_to_states"][config["reverse_code_dict"][x]] for x in l]
        else:
            states = [config["reverse_code_dict"][x] for x in l]
        regions = np.array([region_d[x] for x in states])

        train_ids = list(range(len(regions)))
        test_ids = list(range(len(regions)))

        train_feature_dict = {}
        train_label_dict = {}
        test_feature_dict = {}
        test_label_dict = {}

        for key in range(len(regions)):

            if exclude is None or states[key] not in exclude:
                test_share = int(len(config["clients_label_dict"][key])*region)
                
                train_feature_dict[key] = config["clients_feature_dict"][key][:-test_share]
                train_label_dict[key] = config["clients_label_dict"][key][:-test_share]
                
                test_feature_dict[key] = config["clients_feature_dict"][key][-test_share:]
                test_label_dict[key] = config["clients_label_dict"][key][-test_share:]
        
        if exclude is None:
            config["regions"] = regions
            config["states"] = states

            
    elif region is not None:

        assert region in set(region_d.values()), "Unknown region provided. Available regions: North, Northeast, Center West, Southeast, South"

        l = list(config["clients_feature_dict"].keys())
        states = [config["reverse_code_dict"][x] for x in l]
        regions = np.array([region_d[x] for x in states])

        train_ids = list(np.where(regions != region)[0])
        test_ids = list(np.where(regions == region)[0])

        if frac is not None and frac > 0:
            random.shuffle(train_ids)
            train_ids = train_ids[int(len(train_ids)*frac):]

        train_feature_dict = {key: config["clients_feature_dict"][key] for key in train_ids}
        train_label_dict = {key: config["clients_label_dict"][key] for key in train_ids}

        test_feature_dict = {key: config["clients_feature_dict"][key] for key in test_ids}
        test_label_dict = {key: config["clients_label_dict"][key] for key in test_ids}

    X_train = []
    y_train = []
    X_test = []
    y_test = []
    train_sample_to_client_assignment = []

    cnt = 0
    new_train_feature_dict = {}
    new_train_label_dict = {}
    new_test_feature_dict = {}
    new_test_label_dict = {}        
    for key, features in  train_feature_dict.items():
        labels = train_label_dict[key]
        X_train.extend(features)
        y_train.extend(labels)
        new_train_feature_dict[cnt] = features
        new_train_label_dict[cnt] = labels
        train_sample_to_client_assignment.extend([cnt]*len(labels))
        cnt += 1
        
    if isinstance(region, float):
        cnt = 0
        region = "fraction_"+str(region)

    for key, features in  test_feature_dict.items():
        labels = test_label_dict[key]
        X_test.extend(features)
        y_test.extend(labels)
        new_test_feature_dict[cnt] = features
        new_test_label_dict[cnt] = labels
        cnt += 1

    config["train_sample_to_client_assignment"] = train_sample_to_client_assignment
    config["train_feature_dict"] = new_train_feature_dict
    config["train_label_dict"] = new_train_label_dict
    config["num_train_clients"] = len(new_train_label_dict)
    config["num_test_clients"] = len(new_test_label_dict)
    config["test_feature_dict"] = new_test_feature_dict
    config["test_label_dict"] = new_test_label_dict
    config["X_train"] = X_train
    config["y_train"] = y_train
    config["X_test"] = X_test
    config["y_test"] = y_test
    config["num_train_clients"] = len(train_label_dict)
    config["num_test_clients"] = len(test_label_dict)
    config["train_ids"] = train_ids
    config["test_ids"] = test_ids
    config["test_region"] = str(region)

    return config


# measures all three types of imbalances for data distribution
# takes the config and filename to write to
# returns global label imbalance, global quantity imbalance, dict of local label imbalances, dict of cosine similarities, global feature imbalance, and dict of local feature imbalances
def measure_imbalance(config, filename=None, log=True):
    
    writer = open(filename, 'w') if filename is not None else sys.stdout
    
    local_label_imbalances, global_label_imbalance, local_label_distribution_imbalances = measure_label_imbalance(config)
    global_quantity_imbalance, local_quantity_imbalances = measure_quantity_imbalance(config)
    global_feature_imbalance, local_feature_imbalances = measure_feature_imbalance(config)

    global_cs_median = statistics.median(list(local_label_distribution_imbalances.values()))
    global_cs_stdev = statistics.stdev(list(local_label_distribution_imbalances.values()))
    local_feature_median = statistics.median(list(local_feature_imbalances.values()))
    local_feature_stdev = statistics.stdev(list(local_feature_imbalances.values()))
    
    if log:
        writer.write("\nGlobal Label imbalance "+str(global_label_imbalance))
        writer.write("\nGlobal Label Distribution imbalance Median:"+str(global_cs_median)+", Std.Dev.:"+str(global_cs_stdev))
        writer.write("\nGlobal Quantity imbalance "+str(global_quantity_imbalance))
        writer.write("\nGlobal Feature imbalance "+str(global_feature_imbalance))
        writer.write("\n   Local Feature imbalance Mean:"+str(local_feature_median) + ", Std.Dev.:" + str(local_feature_stdev))
        
        for i in range(config["num_train_clients"]):
            writer.write("\n\nClient "+str(i))
            writer.write("\n  Local Label Imbalance "+str(local_label_imbalances[i]))
            writer.write("\n  Local Label Distribution Imbalance "+str(local_label_distribution_imbalances[i]))
            writer.write("\n  Local Quantity Imbalance "+str(local_quantity_imbalances[i]))
            writer.write("\n  Local Feature Imbalance "+str(local_feature_imbalances[i]))
        
    if filename is not None:
        writer.close()
        
    return global_label_imbalance, local_label_imbalances, global_quantity_imbalance, local_quantity_imbalances, (global_cs_median, global_cs_stdev), local_label_distribution_imbalances, global_feature_imbalance, local_feature_imbalances


# measures local and gloabl imbalance as well as mismatch between class distributions for each client
# takes only the config
# returns dict of local imbalances, the global imbalance and a dict of cosine similarities between class distributions
def measure_label_imbalance(config):
    
    num_quantiles = config["num_quantiles"] if "num_quantiles" in config else 4
    num_classes_analysis = config["num_classes"] if config["num_classes"] > 1 else num_quantiles

    if config["num_classes"] > 1:
        global_counter = Counter(config["y_train"])
    #regression is handled using quartiles
    else:
        quartiles = assign_quantiles(config["y_train"], config["y_train"], num_quantiles)
        global_counter = Counter(quartiles)
    
    global_imbalance = float(max(list(global_counter.values())) / min(list(global_counter.values()))) if len(list(global_counter.values())) > 1 else float("inf")
    global_distribution = [dict(global_counter)[x] if x in dict(global_counter) else 0 for x in range(num_classes_analysis)]
    
    local_imbalances, mismatches = {}, {}
    
    for key, value in config["train_label_dict"].items():
        
        if config["num_classes"] > 1:
            local_counter = Counter(value)
        #regression is handled using quartiles
        else:
            quartiles = assign_quantiles(config["y_train"], value, num_quantiles)
            local_counter = Counter(quartiles)
        
        local_imbalance = float(max(list(local_counter.values()), default=1) / min(list(local_counter.values()), default=1)) if len(list(local_counter.values())) > 1 else float(max(list(local_counter.values())))
        local_imbalances[key] = local_imbalance
        
        local_distribution = [dict(local_counter)[x] if x in dict(local_counter) else 0 for x in range(num_classes_analysis)]
        cos_sim = dot(local_distribution, global_distribution)/(norm(local_distribution)*norm(global_distribution))
        mismatches[key] = 1 - cos_sim

    return local_imbalances, global_imbalance, mismatches


# measures quantity imbalance
# takes only the config
# returns quantity imbalance
def measure_quantity_imbalance(config):
    
    clients_data_count = [len(x) for x in list(config["train_label_dict"].values())]
    global_quantity_imbalance = float(max(clients_data_count) / min(clients_data_count)) if min(clients_data_count) > 0 else float("inf")
    N = sum(clients_data_count) / len(clients_data_count)
    local_quantity_imbalances = {key: len(value)/N for key, value in config["train_label_dict"].items()}
    return global_quantity_imbalance, local_quantity_imbalances
                 
                 
# measures the feature imbalance using the purity metric
# takes only the config
# returns the global feature imbalance and a dict of local feature imbalances
def measure_feature_imbalance(config):
    
    N = sum([len(x) for x in list(config["train_label_dict"].values())]) 
    previous_assignment = np.array(config["train_sample_to_client_assignment"])
    x_k_means = np.array(config["X_train"]).reshape((len(config["X_train"]), -1))

    # compute initial centroids
    centroids = []
    for i in range(config["num_train_clients"]):
        indices = np.where(previous_assignment == i)
        vals = np.array(x_k_means)[indices]
        centroid = np.mean(vals, axis=0)
        centroids.append(centroid)
    # apply k-means
    kmeans = KMeans(n_clusters=config["num_train_clients"], init=np.array(centroids), max_iter=100).fit(x_k_means)
    assignments = np.array(kmeans.labels_)
    assert N == len(assignments), "mismatching lengths of X_train and all clients"

    num_true_assignments = 0
    per_cluster_purity = {}
  
    
    # compute purity
    for i in range(config["num_train_clients"]):
        indices = np.where(assignments == i)
        vals = previous_assignment[indices]
        cluster_label = max(set(list(vals)), key=list(vals).count)  
        true_assignments = sum(1 for j in vals if j == cluster_label)
        num_true_assignments += true_assignments
        per_cluster_purity[i] = true_assignments / len(vals)
    
    purity = num_true_assignments / N
                 
    return purity, per_cluster_purity


# takes a population which is to be devided in quantiles and a list of values
# assignes quantiles to values in accordance with data distribution in population
# returns a list of quantile indices
def assign_quantiles(population, values, q = 4):
    quantiles = []
    for i in range(q+1):
        quantile = np.quantile(population, i/q)
        quantiles.append(quantile)
    quantiles[0] = min(population)-1
    quantiles[-1] = max(population)+1
    ret = []
    for value in values:
        for i, quantile in enumerate(quantiles[:-1]):
            quantile_plus = quantiles[i+1]
            if float(value) >= float(quantile) and float(value) < float(quantile_plus):
                ret.append(i)
                break
    return ret