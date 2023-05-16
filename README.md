# RWSADMM
Mobilizing Personalized Federated Learning in Infrastructure-Less and Heterogeneous Environments via Random Walk Stochastic ADMM

Data heterogeneity in Federated Learning frameworks is addressed in this package named RWSADMM. RWSADMM is personalized FL technique designed for situations where a consistent connection between the central server and all clients cannot be maintained, and data distribution is heterogeneous. To address these challenges, we focus on mobilizing the federated setting, where the server moves between groups of adjacent clients to learn local models.

There exists a graph of clients which collaboratively train a global model. Training is performed locally on clients and aggregation is carried out by a mobilized central server. User can switch between simple random selection of clients and random walk Markov chain by setting the value of parameter "markov_rw" to 0 or 1, respectively. 

There are two main hyperparameters named "beta" and "kappa" which require proper values to obtain optimal performance for RWSADMM. 


This code is part of an ongiong research and the paper is not accepted yet. Please do not use this code in any ways yet. In the case of need for benchmark packages refer to https://github.com/CharlieDinh/pFedMe. 


DataSets:

The datasets need to be generated using the codes in the data folder. There are three folders inside the data folder containing codes for generating MNIST, Cifar10, and Synthetic datasets. 
