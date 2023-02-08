#!/usr/bin/env python
import h5py
import matplotlib.pyplot as plt
import numpy as np
import argparse
import importlib
import random
import os
from FLAlgorithms.servers.serverRWSADMM import RWSADMM
from FLAlgorithms.trainmodel.models import *
from utils.plot_utils import *
import torch
from torchvision import *
import torch.nn as nn
import time
torch.manual_seed(0)


def main(dataset, algorithm, markov_rw, model, batch_size, beta, kappa, lamda, num_glob_iters,
         local_epochs, optimizer, numusers, K, personal_learning_rate, times, gpu):

    # Get device status: Check GPU or CPU
    device = torch.device("cuda:{}".format(gpu) if torch.cuda.is_available() and gpu != -1 else "cpu")

    # added by zp
    start_time = time.time()

    for i in range(times):
        print("---------------Running time:------------",i)
        # Generate model
        if(model == "mclr"):
            if(dataset == "Mnist"):
                model = Mclr_Logistic().to(device), model
            elif (dataset == "Cifar10"):
                model = Mclr_Logistic(3072,10).to(device), model #added by zp
            else:
                model = Mclr_Logistic(60,10).to(device), model
        if(model == "cnn"):
            if(dataset == "Mnist"):
                model = Net().to(device), model
            elif(dataset == "Cifar10"):
                # model = CNNCifar(10).to(device), model
                model = CifarNet().to(device), model
        if(model == "dnn"):
            if(dataset == "Mnist"):
                model = DNN().to(device), model
            elif (dataset == "Cifar10"):
                model = DNN(3072,50, 10).to(device), model # Added by zp 1024*100
            else: 
                model = DNN(60,20,10).to(device), model
        if(model == "resnet"): #added by zp
            net = models.resnet18(pretrained=True) #models is from torchvision library
            numFtrs = net.fc.in_features
            net.fc = nn.Linear(numFtrs, 10)
            model = net, model

        # initializing, training and testing the model
        server = RWSADMM(device, dataset, algorithm, markov_rw, model, batch_size, beta, kappa, lamda, num_glob_iters, local_epochs, optimizer, numusers, K, personal_learning_rate, i)
        server.train()
        server.test()


    # averaging the results of one/several runs of RWSADMM algorithm
    average_data(num_users=numusers, loc_ep1=local_epochs, Numb_Glob_Iters=num_glob_iters, lamb=lamda,
                 beta=beta, kappa=kappa, algorithms="RWSADMM", batch_size=batch_size, dataset=dataset,
                 k=K, personal_learning_rate=personal_learning_rate, times=times)
    # averaging the results of one/several runs of RWSADMM_p algorithm
    # average_data(num_users=numusers, loc_ep1=local_epochs, Numb_Glob_Iters=num_glob_iters, lamb=lamda,
    #              beta=beta, kappa=kappa, algorithms="RWSADMM_p", batch_size=batch_size, dataset=dataset,
    #              k=K, personal_learning_rate=personal_learning_rate, times=times)


    # added by zp
    finish_time = time.time()
    time_diff = finish_time - start_time
    print("---------------------------------")
    print("The elapsed duration is: ", "{:.2f}".format(time_diff), "seconds. \n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Cifar10", choices=["Mnist", "Synthetic", "Cifar10"])
    parser.add_argument("--model", type=str, default="mclr", choices=["dnn", "mclr", "cnn","resnet"])
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--beta", type=float, default=10, help="Beta parameter for RWSADMM")
    parser.add_argument("--kappa", type=float, default=0.001, help="Kappa parameter for RWSADMM")
    parser.add_argument("--lamda", type=int, default=20, help="Regularization term")
    parser.add_argument("--num_global_iters", type=int, default=5)
    parser.add_argument("--local_epochs", type=int, default=3)
    parser.add_argument("--optimizer", type=str, default="SGD")
    parser.add_argument("--algorithm", type=str, default="RWSADMM")
    parser.add_argument("--markov_rw", type=int, default = 1, choices=[1,0]) # 1 for random walk markov, 0 simple random selection
    parser.add_argument("--numusers", type=int, default=5, help="Number of Users per round")
    parser.add_argument("--K", type=int, default=5, help="Computation steps")
    parser.add_argument("--personal_learning_rate", type=float, default=0.01, help="Persionalized learning rate to caculate theta aproximately using K steps")
    parser.add_argument("--times", type=int, default=1, help="running time")
    parser.add_argument("--gpu", type=int, default=0, help="Which GPU to run the experiments, -1 mean CPU, 0,1,2 for GPU")
    args = parser.parse_args()

    print("=" * 80)
    print("Summary of training process:")
    print("Algorithm: {}".format(args.algorithm))
    print("Batch size: {}".format(args.batch_size))
    print("Beta parameter       : {}".format(args.beta))
    print("Kappa parameter       : {}".format(args.kappa))
    print("Subset of users      : {}".format(args.numusers))
    print("Number of global rounds       : {}".format(args.num_global_iters))
    print("Number of local rounds       : {}".format(args.local_epochs))
    print("Dataset       : {}".format(args.dataset))
    print("Local Model       : {}".format(args.model))
    print("=" * 80)

    main(
        dataset=args.dataset,
        algorithm = args.algorithm,
        markov_rw = args.markov_rw,
        model=args.model,
        batch_size=args.batch_size,
        beta=args.beta,
        kappa= args.kappa,
        lamda = args.lamda,
        num_glob_iters=args.num_global_iters,
        local_epochs=args.local_epochs,
        optimizer= args.optimizer,
        numusers = args.numusers,
        K=args.K,
        personal_learning_rate=args.personal_learning_rate,
        times = args.times,
        gpu=args.gpu
        )
