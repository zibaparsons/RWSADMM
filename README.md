# RWSADMM
Mobilizing Federated Networks via Random Walk Stochastic ADMM

Data heterogeneity in Federated Learning frameworks is addressed in this package named RWSADMM. RWSADMM is personalized FL technique designed for situations where a consistent connection between the central server and all clients cannot be maintained, and data distribution is heterogeneous. To address these challenges, we focus on mobilizing the federated setting, where the server moves between groups of adjacent clients to learn local models.

There exists a graph of clients which collaboratetively train a global model. Training is performed locally on clients and aggregation is carried out by a mobilized central server. User can switch between simple random selection of clients and random walk Markov chain by setting the value of parameter "markov_rw" to 0 or 1, respectively. 

There are two main hyperparameters named "beta" and "kappa" which require proper values to obtain optimal performance for RWSADMM. 
