import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
from torch.utils.data import DataLoader
import numpy as np
import copy

class User:
    """
    Base class for users in federated learning.
    """
    def __init__(self, device, id, train_data, test_data, model, batch_size = 0, learning_rate = 0, beta = 0 , lamda = 0, local_epochs = 0):

        self.device = device
        self.model_y = copy.deepcopy(model) # global y
        self.model = copy.deepcopy(model)
        self.id = id  # integer
        self.train_samples = len(train_data)
        self.test_samples = len(test_data)
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.beta = beta
        self.lamda = lamda
        self.local_epochs = local_epochs
        self.trainloader = DataLoader(train_data, self.batch_size)
        self.testloader =  DataLoader(test_data, self.batch_size)
        self.testloaderfull = DataLoader(test_data, self.test_samples)
        self.trainloaderfull = DataLoader(train_data, self.train_samples)
        self.iter_trainloader = iter(self.trainloader)
        self.iter_testloader = iter(self.testloader)
        self.local_model = copy.deepcopy(list(self.model.parameters()))
        # those parameters are for persionalized federated learing.
        self.local_model_x = copy.deepcopy(model)
        for param in self.local_model_x.parameters():
            param.data = torch.zeros_like(param.data)
        # self.local_model_x_old = copy.deepcopy(model)
        # for param in self.local_model_x_old.parameters():
        #     param.data = torch.zeros_like(param.data)
        self.local_dual_z = copy.deepcopy(model)
        for param in self.local_dual_z.parameters():
            param.data = torch.zeros_like(param.data)
        # self.local_dual_z_old = copy.deepcopy(model)
        # for param in self.local_dual_z_old.parameters():
        #     param.data = torch.zeros_like(param.data)

        self.persionalized_model = copy.deepcopy(list(self.model.parameters()))
        self.persionalized_model_bar = copy.deepcopy(list(self.model.parameters()))
    
    def set_parameters(self, model):
        for old_param, new_param, local_param in zip(self.model.parameters(), model.parameters(), self.local_model):
            old_param.data = new_param.data.clone()
            local_param.data = new_param.data.clone()
        #self.local_weight_updated = copy.deepcopy(self.optimizer.param_groups[0]['params'])

    def get_parameters(self):
        for param in self.model.parameters():
            param.detach()
        return self.model.parameters()
    
    def clone_model_paramenter(self, param, clone_param):
        for param, clone_param in zip(param, clone_param):
            clone_param.data = param.data.clone()
        return clone_param
    
    def get_updated_parameters(self):
        return self.local_weight_updated
    
    def update_parameters(self, new_params):
        for param , new_param in zip(self.model.parameters(), new_params):
            param.data = new_param.data.clone()

    def get_gradget_parameterss(self):
        grads = []
        for param in self.model.parameters():
            if param.grad is None:
                grads.append(torch.zeros_like(param.data))
            else:
                grads.append(param.grad.data)
        return grads

    def test(self):
        self.local_model_x.eval()
        test_acc = 0
        for x, y in self.testloaderfull:
            x, y = x.to(self.device), y.to(self.device)
            output = self.local_model_x(x)
            test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
            #@loss += self.loss(output, y)
            #print(self.id + ", Test Accuracy:", test_acc / y.shape[0] )
            #print(self.id + ", Test Loss:", loss)
        return test_acc, y.shape[0]

    def train_error_and_loss(self):
        self.local_model_x.eval()
        train_acc = 0
        loss = 0
        for x, y in self.trainloaderfull:
            x, y = x.to(self.device), y.to(self.device)
            output = self.local_model_x(x)
            train_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
            loss += self.loss(output, y)
            #print(self.id + ", Train Accuracy:", train_acc)
            #print(self.id + ", Train Loss:", loss)
        return train_acc, loss , self.train_samples
    
    def test_persionalized_model(self):
        self.model.eval()
        test_acc = 0
        self.update_parameters(self.persionalized_model_bar)
        for x, y in self.testloaderfull:
            x, y = x.to(self.device), y.to(self.device)
            output = self.local_model_x(x)
            test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
        self.update_parameters(self.local_model)
        return test_acc, y.shape[0]

    def train_error_and_loss_persionalized_model(self):
        self.model.eval()
        train_acc = 0
        loss = 0
        self.update_parameters(self.persionalized_model_bar)
        for x, y in self.trainloaderfull:
            x, y = x.to(self.device), y.to(self.device)
            output = self.local_model_x(x)
            train_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
            loss += self.loss(output, y)
        self.update_parameters(self.local_model)
        return train_acc, loss , self.train_samples
    
    def get_next_train_batch(self):
        try:
            # Samples a new batch for persionalizing
            (X, y) = next(self.iter_trainloader)
        except StopIteration:
            # restart the generator if the previous generator is exhausted.
            self.iter_trainloader = iter(self.trainloader)
            (X, y) = next(self.iter_trainloader)
        return (X.to(self.device), y.to(self.device))
    
    def get_next_test_batch(self):
        try:
            # Samples a new batch for persionalizing
            (X, y) = next(self.iter_testloader)
        except StopIteration:
            # restart the generator if the previous generator is exhausted.
            self.iter_testloader = iter(self.testloader)
            (X, y) = next(self.iter_testloader)
        return (X.to(self.device), y.to(self.device))

    def save_model(self):
        model_path = os.path.join("models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.model, os.path.join(model_path, "user_" + self.id + ".pt"))

    def load_model(self):
        model_path = os.path.join("models", self.dataset)
        self.model = torch.load(os.path.join(model_path, "server" + ".pt"))
    
    @staticmethod
    def model_exists():
        return os.path.exists(os.path.join("models", "server" + ".pt"))

    def update_X(self):
        X = ()
        for param in self.model.named_parameters():
            X += (param.detach().cpu().clone(),)
        return X

    def update_Z(X, U, args):
        new_Z = ()
        for x, u in zip(X, U):
            z = x + u
        return new_Z

    def update_Z_l1(X, U, args):
        new_Z = ()
        delta = args.alpha / args.rho
        for x, u in zip(X, U):
            z = x + u
            new_z = z.clone()
            if (z > delta).sum() != 0:
                new_z[z > delta] = z[z > delta] - delta
            if (z < -delta).sum() != 0:
                new_z[z < -delta] = z[z < -delta] + delta
            if (abs(z) <= delta).sum() != 0:
                new_z[abs(z) <= delta] = 0
            new_Z += (new_z,)
        return new_Z

    def update_U(U, X, Z):
        new_U = ()
        for u, x, z in zip(U, X, Z):
            new_u = u + x - z
            new_U += (new_u,)
        return new_U