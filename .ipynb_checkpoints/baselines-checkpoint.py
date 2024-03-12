import numpy as np
import sys
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, mean_absolute_error, mean_squared_error
from tqdm import tqdm
from copy import deepcopy
import model_utils
import models
import load_data
import strategies
from copy import deepcopy
from collections import Counter
import scikitplot as skplt
import matplotlib.pyplot as plt
import shap
import pandas as pd
import seaborn as sns
from math import log2
from k_means_constrained import KMeansConstrained


# perform leave-one-(client/cluster)-out to measure the contribution on performance
def measure_contribution(central_model, central_f1s, config, per_client_imbalances, filename=None, log_per_round=False, plot_clustering=False):

    if "average_lxo" not in config:
        raise KeyError("\"average_lxo\" is not defined")

    writer = open(filename, 'w') if filename is not None else sys.stdout
    clients_feature_dict = config["clients_feature_dict"]
    clients_label_dict = config["clients_label_dict"]
    max_idx = config["num_clients"] - 1

    per_client_imbalances = np.array(per_client_imbalances)

    X_test = torch.tensor(config["X_test"]).float()

    leave_out_performances = []
    cluster_imbalances = []
    influences = []
    reputations = []

    _, pred_central = model_utils.perform_inference(central_model, X_test, config["batch_size"], config["device"], is_bert=config["model_name"].lower() == "bert")

    if config["average_lxo"] <= 1:

        leave_out_clusters = [[key] for key in clients_feature_dict]
        inverted_leave_out_clusters = []
        for key in clients_feature_dict:
            a = list(range(len(clients_feature_dict)))
            del a[key] 
            inverted_leave_out_clusters.append(a)

    elif config["average_lxo"] > 1:

        leave_out_clusters = []
        inverted_leave_out_clusters = []
        num_clusters = round((config["num_clients"]) / config["average_lxo"])
        min_cluster_size = round(config["average_lxo"] / 2)
        max_cluster_size = round(config["average_lxo"] * 2)
        
        fit_on_client_imbalances = per_client_imbalances
        kmeans = KMeansConstrained(n_clusters=num_clusters, size_min= min_cluster_size, size_max=max_cluster_size, max_iter=300).fit(fit_on_client_imbalances)
        assignments = np.array(kmeans.labels_)

        for n in range(num_clusters):
            inverted_indices = np.where(assignments != n)[0].tolist()
            indices = np.where(assignments == n)[0].tolist()
            if len(indices) < config["num_clients"]:
                leave_out_clusters.append(indices)
                inverted_leave_out_clusters.append(inverted_indices)

    writer.write("\n\nNumber of clients per cluster: " + str([len(x) for x in leave_out_clusters]) + "\n")

    for index, cluster in enumerate(inverted_leave_out_clusters):

        if log_per_round:
            writer.write("\n\nIteration "+str(index))

        tmp_clients_feature_dict = {}
        tmp_clients_label_dict = {}

        for i, key in enumerate(cluster):
            tmp_clients_feature_dict[i] = clients_feature_dict[key]
            tmp_clients_label_dict[i] = clients_label_dict[key]

        config["num_clients"] = len(cluster)
        config["clients_feature_dict"] = tmp_clients_feature_dict
        config["clients_label_dict"] = tmp_clients_label_dict

        learning_strategy = strategies.get_strategy_by_name(config)
        model, lxo_f1s = learning_strategy.run(config, filename=filename, log_per_round=False, return_f1s=True)

        # compute cluster influence
        _, pred_lxo = model_utils.perform_inference(model, X_test, config["batch_size"], config["device"], is_bert=config["model_name"].lower() == "bert")
        influence = np.sum(np.sqrt(np.sum(np.square(np.subtract(pred_central, pred_lxo)), axis=1))) / len(pred_central)
        influences.append(influence)
        
        #compute cluster reputation 
        assert config["rounds"] >= config["reputation_ts"], "'reputation_ts' must be smaller than 'rounds'"
        rounds_offset = config["rounds"] / config["reputation_ts"]
        offset_mlt = 1
        reputation = []
        for i, central_f1 in enumerate(central_f1s):
            if (i+1) == round(rounds_offset*offset_mlt):
                r = 1 if central_f1 >= lxo_f1s[i] else 0
                reputation.append(r)
                offset_mlt += 1
        reputations.append(sum(reputation) / len(reputation))

        acc, pre, rec, f1, _ = evaluate_minority(model, config["X_test"], config["y_test"], config, filename=None, log=False)
        leave_out_performances.append((acc, pre, rec, f1))

        mean_imbalances = np.mean(per_client_imbalances[leave_out_clusters[index]], axis=0).tolist()

        cluster_imbalances.append(mean_imbalances)

    if filename is not None:
        writer.close()

    config["num_clients"] = max_idx + 1
    config["clients_feature_dict"] = clients_feature_dict
    config["clients_label_dict"] = clients_label_dict

    return leave_out_performances, influences, reputations, cluster_imbalances, leave_out_clusters, None


# computes accuracy, precision, recall and f1 for given model and instances
def evaluate(model, X_test, y_test, config, filename=None, title="", log=True):    

    writer = open(filename, 'w') if filename is not None else sys.stdout
    if log:
        writer.write("\n"+str(title))
    
    X_test = torch.tensor(X_test).float()
    y_test = torch.tensor(y_test)

    predictions, probabilities = model_utils.perform_inference(model, X_test, config["batch_size"], config["device"], is_bert=False)
    
    count = Counter(y_test.tolist())
    true_dict = {}
    for x in list(range(config["num_classes"])):
        if x not in count:
            true_dict[x] = 0
        else:
            true_dict[x] = count[x] / len(y_test)
    t_percentage_count = [(i, true_dict[i]) for i in true_dict]

    if log:
        writer.write("\n True Label Distribution "+ str(t_percentage_count))
    count = Counter(predictions)
    pred_dict = {}
    for x in list(range(config["num_classes"])):
        if x not in count:
            pred_dict[x] = 0
        else:
            pred_dict[x] = count[x] / len(predictions)
    p_percentage_count = [(i, pred_dict[i]) for i in pred_dict]
    if log:
        writer.write("\n Predicted Label Distribution " + str(p_percentage_count))
    pre, rec, f1, _ = precision_recall_fscore_support(y_test, predictions, average=config["evaluation_averaging"])
    pres, recs, f1s, _ = precision_recall_fscore_support(y_test, predictions, average=None, labels=list(range(config["num_classes"])))
    acc = np.sum(np.array(predictions) == y_test.numpy()) / len(y_test)

    if log:
        writer.write("\n Accuracy: "+str(acc))
        writer.write("\n Precision: "+str(pre) + " " + str(pres))
        writer.write("\n Recall: "+str(rec) + " " + str(recs))
        writer.write("\n F1-Score: "+str(f1) + " " + str(f1s))

    if filename is not None:
        writer.close()
    return acc, pre, rec, f1, [(true_dict[x],f1s[x]) for x in list(range(config["num_classes"]))]
