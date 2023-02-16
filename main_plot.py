#!/usr/bin/env python
import h5py
import matplotlib.pyplot as plt
import numpy as np
import argparse
import importlib
import random
import os
from utils.plot_utils import *
import torch
torch.manual_seed(1)

if(1): # plot for beta tune
    numusers = 5
    num_glob_iters = 200
    dataset = "Cifar10"
    model_str = "Strongly convex MLR" # Non-convex MLP, Non-convex CNN
    local_ep = [5,5,5,5,5]
    lamda = [30,30,30,30,30]
    beta = [0.1, 1, 10, 100, 1000]
    kappa =  [0.001, 0.001, 0.001, 0.001, 0.001]
    batch_size = [20,20,20,20,20]
    K = [5,5,5,5,5]
    personal_learning_rate = [0.01,0.01,0.01,0.01,0.01]
    algorithms = ["RWSADMM_p", "RWSADMM_p", "RWSADMM_p", "RWSADMM_p", "RWSADMM_p"]
    plot_summary_one_figure_Beta(num_users=numusers, loc_ep1=local_ep, Numb_Glob_Iters=num_glob_iters, lamb=lamda,
                                          beta=beta, kappa= kappa, algorithms_list=algorithms, batch_size=batch_size, dataset=dataset, k = K, personal_learning_rate = personal_learning_rate,
                                         model_str=model_str)

if(0): # plot for Synthetic covex
    numusers = 20
    num_glob_iters = 800
    dataset = "Synthetic" #"Cifar10"
    local_ep = [20,20,20,20]
    lamda = [15,15,15,15]#[20,20,20,20]
    learning_rate = [0.01, 0.01,0.01,0.01]#[0.005, 0.005, 0.005, 0.005]
    beta =  [0.01, 0.01,1.0,1.0]#[1.0, 1.0, 0.001, 1.0]
    batch_size = [20,20,20,20]
    K = [5,5,5,5]
    personal_learning_rate = [0.01,0.01,0.01,0.01]

    algorithms = ["RWSADMM","pFedMe_p"] #[FedMe_p","RWSADMM","FedAvg","PerAvg_p"]
    plot_summary_one_figure_synthetic_Compare(num_users=numusers, loc_ep1=local_ep, Numb_Glob_Iters=num_glob_iters, lamb=lamda,
                               learning_rate=learning_rate, beta = beta, algorithms_list=algorithms, batch_size=batch_size, dataset=dataset, k = K, personal_learning_rate = personal_learning_rate)
    # plot_summary_one_figure_mnist_Beta(num_users=numusers, loc_ep1=local_ep, Numb_Glob_Iters=num_glob_iters, lamb=lamda,
    #                            learning_rate=learning_rate, beta = beta, algorithms_list=algorithms, batch_size=batch_size, dataset=dataset, k=K, personal_learning_rate = personal_learning_rate)
