# RWSADMM
Mobilizing Personalized Federated Learning in Infrastructure-Less and Heterogeneous Environments via Random Walk Stochastic ADMM

This package is the source code for RWSADMM, proposed in the above mentioned paper, and accepted and presented at the Thirty-seventh Conference on Neural Information Processing Systems (NeurIPS 2023). Please cite our paper using the following. 

  - Parsons, Z., Dou, F., Du, H., Song, Z., & Lu, J. (2024). Mobilizing personalized federated learning in infrastructure-less and heterogeneous environments via random walk stochastic ADMM. Advances in Neural Information Processing Systems, 36.

Data heterogeneity in Federated Learning frameworks is addressed in this package named RWSADMM. RWSADMM is a personalized FL technique designed for situations where a consistent connection between the central server and all clients cannot be maintained, and data distribution is heterogeneous. To address these challenges, we focus on mobilizing the federated setting, where the server moves between groups of adjacent clients to learn local models.

There exists a graph of clients who collaboratively train a global model. Training is performed locally on clients, and a mobilized central server performs the aggregation. The user selection technique can be a simple random selection of clients or a random walk Markov chain by setting the value of parameter "markov_rw" to 0 or 1, respectively. 

There are two primary hyperparameters named "beta" and "kappa" which require proper values to obtain optimal performance for RWSADMM. 


DataSets:

The datasets need to be generated using the codes in the data folder. Three folders inside the data folder contain codes for generating MNIST, Cifar10, and Synthetic datasets. 
The data is also benchmarked in the reference GitHub package: https://github.com/CharlieDinh/pFedMe. 
