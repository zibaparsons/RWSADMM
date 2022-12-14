import torch
import os

from FLAlgorithms.users.userRWSADMM import UserRWSADMM
from FLAlgorithms.servers.serverbase import Server
from utils.model_utils import read_data, read_user_data
import numpy as np

import time

# Implementation for RWSADMM Server

class RWSADMM(Server):
    def __init__(self, device,  dataset, algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters,
                 local_epochs, optimizer, num_users, K, personal_learning_rate, times):
        super().__init__(device, dataset,algorithm, model[0], batch_size, learning_rate, beta, lamda, num_glob_iters,
                         local_epochs, optimizer, num_users, times)

        # Initialize data for all  users
        data = read_data(dataset)
        total_users = len(data[0])
        self.K = K
        self.personal_learning_rate = personal_learning_rate
        for i in range(total_users):
            id, train , test = read_user_data(i, data, dataset)
            user = UserRWSADMM(device, id, train, test, model, batch_size, learning_rate, beta, lamda, local_epochs, optimizer, K, personal_learning_rate)
            self.users.append(user)
            self.total_train_samples += user.train_samples
        print("Number of users / total users:",num_users, " / " ,total_users)
        print("Finished creating RWSADMM server.")

    def send_grads(self):
        assert (self.users is not None and len(self.users) > 0)
        grads = []
        for param in self.model.parameters():
            if param.grad is None:
                grads.append(torch.zeros_like(param.data))
            else:
                grads.append(param.grad)
        for user in self.users:
            user.set_grads(grads)

    def train(self):
        loss = []
        for glob_iter in range(self.num_glob_iters):
            print("-------------Round number: ",glob_iter, " -------------")
            # send all parameter for users 
            self.send_parameters()

            st_time = time.time()  # added by zp
            # Evaluate global model on user for each iteration
            print("Evaluate global model")
            print("")
            self.evaluate()

            # do update for all users not only selected users
            for user in self.users:
                user.train(self.local_epochs, self.num_users) #* user.train_samples

            # choose several users to send back updated model to server
            # self.personalized_evaluate()
            self.selected_users = self.select_users(glob_iter,self.num_users)

            # Evaluate global model on user for each iteration
            self.evaluate_personalized_model()
            #self.aggregate_parameters()
            self.persionalized_aggregate_parameters()

            # # added by zp
            # fin_time = time.time()
            # diff_2 = fin_time - st_time
            # print("---------------------------------")
            # print("The elapsed duration for 1 iteration is: ", "{:.2f}".format(diff_2), "seconds. \n")

        #print(loss)
        self.save_results()
        self.save_model()
