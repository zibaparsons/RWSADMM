import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
from torch.utils.data import DataLoader
from FLAlgorithms.optimizers.fedoptimizer import pFedMeOptimizer
from FLAlgorithms.users.userbase import User
import copy
import time

# Implementation for RWSADMM clients

class UserRWSADMM(User):
    def __init__(self, device, numeric_id, train_data, test_data, model, batch_size, beta, kappa, lamda,
                 local_epochs, optimizer, K, personal_learning_rate):
        super().__init__(device, numeric_id, train_data, test_data, model[0], batch_size, beta, kappa, lamda,
                         local_epochs)
        if(model[1] == "Mclr_CrossEntropy"):
            self.loss = nn.CrossEntropyLoss()
        else:
            self.loss = nn.NLLLoss()
        self.K = K
        self.personal_learning_rate = personal_learning_rate
        self.optimizer = pFedMeOptimizer(self.model.parameters(), lr=self.personal_learning_rate, lamda=self.lamda)

    def set_grads(self, new_grads):
        if isinstance(new_grads, nn.Parameter):
            for model_grad, new_grad in zip(self.model.parameters(), new_grads):
                model_grad.data = new_grad.data
        elif isinstance(new_grads, list):
            for idx, model_grad in enumerate(self.model.parameters()):
                model_grad.data = new_grads[idx]

    def train(self, epochs, num_users):
        eps = 1e-5
        start_time = time.time()
        self.model.train()
        for epoch in range(1, self.local_epochs + 1):  # local update
            self.model.train()
            X, y = self.get_next_train_batch()
            # K is number of personalized steps
            for i in range(self.K):
                self.local_model_x.train()
                local_model_x_old = copy.deepcopy(self.local_model_x)
                local_dual_z_old = copy.deepcopy(self.local_dual_z)
                self.optimizer.zero_grad()
                output = self.local_model_x(X)
                loss = self.loss(output, y)
                loss.backward()
                for param_x, param_x_old, param_z_old, param_y, param_z in zip(self.local_model_x.parameters(), \
                                                                               local_model_x_old.parameters(),
                                                                               local_dual_z_old.parameters(), \
                                                                               self.model.parameters(),
                                                                               self.local_dual_z.parameters()):
                    # param_x_old.data = param_x.data
                    # param_z_old.data = param_z.data
                    param_x.data = param_y.data - (1/self.beta) * (param_x.grad.data - param_z.data + eps) #* param_x.grad.data #
                    param_z.data = param_z.data + self.kappa*self.beta * (param_x.data - param_y.data - eps)
                    param_y.data = param_y.data + 1 / num_users * (
                            param_x.data - (1/self.beta)*param_z.data - param_x_old.data + (1/self.beta)*param_z_old.data)
            self.kappa = 0.99 * self.kappa
            persionalized_model_bar = copy.copy(self.local_model_x)

            # update local weight after finding approximate theta
            for new_param, localweight in zip(persionalized_model_bar.parameters(), self.local_model_x.parameters()):
                localweight.data = localweight.data - self.lamda * (1/self.beta) * (localweight.data - new_param.data)

        #update local model as local_weight_upated
        self.update_parameters(self.local_model_x.parameters())

        # recording the training times
        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time
