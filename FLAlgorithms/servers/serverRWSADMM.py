import torch
import networkx as nx
import numpy as np
from FLAlgorithms.users.userRWSADMM import UserRWSADMM
from FLAlgorithms.servers.serverbase import Server
from utils.model_utils import read_data, read_user_data
import time

# Implementation for RWSADMM Server

class RWSADMM(Server):
    def __init__(self, device, dataset, algorithm, markov_rw, model, batch_size, beta, kappa, lamda, num_glob_iters,
                 local_epochs, optimizer, num_users, K, personal_learning_rate, times):
        super().__init__(device, dataset, algorithm, model[0], batch_size, beta, kappa, lamda, num_glob_iters,
                         local_epochs, optimizer, num_users, times)

        # Initialize data for all  users
        data = read_data(dataset)
        self.total_users = len(data[0])
        self.num_users = num_users # added for dynamic graph update in train method
        self.K = K
        self.personal_learning_rate = personal_learning_rate
        self.G = self.graph_users(self.total_users, num_users)
        self.markov_rw = markov_rw # settings to apply random walk MC or simple random selection
        self.graph_update_frequency = 10
        for i in range(self.total_users):
            id, train , test = read_user_data(i, data, dataset)
            user = UserRWSADMM(device, id, train, test, model, batch_size, beta, kappa, lamda, local_epochs, optimizer, K, personal_learning_rate)
            self.users.append(user)
            self.total_train_samples += user.train_samples
        print("Number of users / total users:",num_users, " / " ,self.total_users)
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
            if (self.markov_rw == 1): # applying RW-MC or random selection
                self.selected_users = self.select_users_markov(glob_iter, self.num_users)
            else:
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

            # dynamic graph update
            if (glob_iter>0 & (glob_iter%self.graph_update_frequency)==0):
                self.G = self.graph_users(self.total_users, self.num_users)
        #print(loss)
        self.save_results()
        self.save_model()

    def graph_users(self, total_users, num_users):
        G = nx.Graph()
        G.add_nodes_from(range(total_users))
        for node in G.nodes():
            neighbors = np.random.choice(list(G.nodes()), num_users, replace=False)
            for neighbor in neighbors:
                G.add_edge(node, neighbor)
        return G

    def select_users_markov(self, glob_iter, num_users):
        # Create transition matrix T
        T = np.zeros((self.total_users, self.total_users))
        for i in range(self.total_users):
            neighbors = list(self.G.neighbors(i))
            for neighbor in neighbors:
                T[i][neighbor] = 1/len(neighbors)
        # perform Markov random walk
        current_center = np.random.choice(range(self.total_users))
        for i in range(glob_iter):
            current_center = np.random.choice(range(self.total_users), p=T[current_center])
        # select current center and its neighbors as selected users
        selected_users = list(self.G.neighbors(current_center))
        selected_users.append(current_center)
        return (self.users[i] for i in selected_users)
