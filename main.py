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

torch.cuda.empty_cache()
# import gc
# del vars
# gc.collect()
# # The flag below controls whether to allow TF32 on matmul. This flag defaults to False
# # in PyTorch 1.12 and later.
# torch.backends.cuda.matmul.allow_tf32 = True
# # The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
# torch.backends.cudnn.allow_tf32 = True

# torch.backends.cuda.matmul.allow_tf32 = False
# torch.backends.cudnn.allow_tf32 = False
# import os
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:<enter-size-here>"
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"

# os.environ["PYTORCH_CUDA_ALLOC_CONF"]="max_split_size_mb:100"

# clearing gpu cache 
# from GPUtil import showUtilization as gpu_usage
# from numba import cuda

# def free_gpu_cache():
#     print("Initial GPU Usage")
#     gpu_usage()                             

#     torch.cuda.empty_cache()

#     cuda.select_device(0)
#     cuda.close()
#     cuda.select_device(0)

#     print("GPU Usage after emptying the cache")
#     gpu_usage()

# free_gpu_cache()  



def main(dataset, algorithm, markov_rw, model, batch_size, beta, kappa, lamda, num_glob_iters,
         local_epochs, optimizer, numusers, K, personal_learning_rate, times, gpu):
    # time_list = []
    # Get device status: Check GPU or CPU
    device = torch.device("cuda:{}".format(gpu) if torch.cuda.is_available() and gpu != -1 else "cpu")
    # device = "cpu"
    # added by ZibaP
    time_list = []

    for i in range(times):
        print("---------------Running time:------------",i)

        # added by zp
        start_time = time.time()


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
                # model = CifarNet().to(device), model
                model = Net().to(device), model

        if(model == "largerCNN"):
            if(dataset == "Mnist"):
                model = largerNet().to(device), model
            elif(dataset == "Cifar10"):
                # model = CNNCifar(10).to(device), model
                model = largerNet().to(device), model

        if(model == "megaCNN"):
            if(dataset == "Mnist"):
                model = megaNet().to(device), model
            elif(dataset == "Cifar10"):
                # model = CNNCifar(10).to(device), model
                model = megaNet().to(device), model


        # if model == "cnn":
        #     if dataset == "Mnist":
        #         model = MiniFedAvgCNN(in_features=1, num_classes=10, dim=1024).to(device), model
        #     elif dataset == "Cifar10":
        #         model = FedAvgCNN(in_features=3, num_classes=10, dim=1600).to(device), model
        #     else:
        #         model = FedAvgCNN(in_features=3, num_classes=10, dim=10816).to(device), model

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

        if(model == "alexnet"): #added by zibap 3/17/2023
            net = models.AlexNet(num_classes=10, dropout=0.5) #number of classes = 10
            # net = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet',pretrained=True)
            # numFtrs = net.fc.in_features
            # net.fc = nn.Linear(numFtrs, 10)
            net.classifier[6] = nn.Linear(4096,10)
            model = net.to(device), model


        # initializing, training and testing the model
        server = RWSADMM(device, dataset, algorithm, markov_rw, model, batch_size, beta, kappa, lamda, num_glob_iters, local_epochs, optimizer, numusers, K, personal_learning_rate, i)
        server.train()
        server.test()

        # added by ZibaP
        time_list.append(time.time()-start_time)

    # added by ZibaP
    print(f"\nAverage time cost: {round(np.average(time_list), 2)}s.")


    # averaging the results of one/several runs of RWSADMM algorithm
    average_data(num_users=numusers, loc_ep1=local_epochs, Numb_Glob_Iters=num_glob_iters, lamb=lamda,
                 beta=beta, kappa=kappa, algorithms="RWSADMM", batch_size=batch_size, dataset=dataset,
                 k=K, personal_learning_rate=personal_learning_rate, times=times)
    # averaging the results of one/several runs of RWSADMM_p algorithm
    average_data(num_users=numusers, loc_ep1=local_epochs, Numb_Glob_Iters=num_glob_iters, lamb=lamda,
                 beta=beta, kappa=kappa, algorithms="RWSADMM_p", batch_size=batch_size, dataset=dataset,
                 k=K, personal_learning_rate=personal_learning_rate, times=times)


    # added by zp
    # finish_time = time.time()
    # time_diff = finish_time - start_time
    # print("---------------------------------")
    # print("The elapsed duration is: ", "{:.2f}".format(time_diff), "seconds. \n")

# best setting for Cifar10 beta=100 kappa=0.0001, K=3

# best setting for Mnist for now beta=100 and kappa=0.001
# beta=10 kappa=0.01 was not good; neither beta=1000 kappa=0.001;
# beta=100 kappa=0.01 is not as good as the best
# with beta=10 and kappa=0.001, i got 98.28%


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Cifar10", choices=["Mnist", "Synthetic", "Cifar10"])
    parser.add_argument("--model", type=str, default="megaCNN", choices=["dnn", "mclr", "cnn","largerCNN","resnet","alexnet"])
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--beta", type=float, default=100, help="Beta parameter for RWSADMM") # optimal value was 100
    parser.add_argument("--kappa", type=float, default=0.001, help="Kappa parameter for RWSADMM") # optimal value was 0.001
    parser.add_argument("--lamda", type=int, default=20, help="Regularization term")
    parser.add_argument("--num_global_iters", type=int, default=200)
    parser.add_argument("--local_epochs", type=int, default=20)
    parser.add_argument("--optimizer", type=str, default="SGD")
    parser.add_argument("--algorithm", type=str, default="RWSADMM")
    parser.add_argument("--markov_rw", type=int, default = 1, choices=[1,0]) # 1 for random walk markov, 0 simple random selection
    parser.add_argument("--numusers", type=int, default=10, help="Number of Users per round")
    parser.add_argument("--K", type=int, default=5, help="Personalized Computation steps") # higher number more personalization
    parser.add_argument("--personal_learning_rate", type=float, default=0.01, help="Persionalized learning rate to caculate theta aproximately using K steps")
    parser.add_argument("--times", type=int, default=1, help="running time")
    parser.add_argument("--gpu", type=int, default=1, help="Which GPU to run the experiments, -1 mean CPU, 0,1,2 for GPU")
    args = parser.parse_args()

    print("=" * 80)
    print("Summary of training process:")
    print("Algorithm: {}".format(args.algorithm))
    print("Batch size: {}".format(args.batch_size))
    print("Beta parameter       : {}".format(args.beta))
    print("Kappa parameter       : {}".format(args.kappa))
    print("Lambda parameter      : {}".format(args.lamda))
    print("Subset of users      : {}".format(args.numusers))
    print("Number of global rounds       : {}".format(args.num_global_iters))
    print("Number of local rounds       : {}".format(args.local_epochs))
    print("K steps         : {}".format(args.K))
    print("Personalized learning rate   : {}".format(args.personal_learning_rate))
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
