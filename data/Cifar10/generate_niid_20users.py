from sklearn.datasets import fetch_openml
from tqdm import trange
import numpy as np
import random
import json
import os
import torch
import torchvision
import torchvision.transforms as transforms

# random.seed(1)
# np.random.seed(1)
# NUM_USERS = 50 # should be muitiple of 10
# NUM_LABELS = 2
# # Setup directory for train/test data
# train_path = './data/Mnist/data/train/mnist_train_noniid.json'
# test_path = './data/Mnist/data/test/mnist_test_noniid.json'
# dir_path = os.path.dirname(train_path)
# if not os.path.exists(dir_path):
#     os.makedirs(dir_path)
# dir_path = os.path.dirname(test_path)
# if not os.path.exists(dir_path):
#     os.makedirs(dir_path)

# # Get MNIST data, normalize, and divide by level
# mnist = fetch_openml('mnist_784', data_home='./data')
# mu = np.mean(mnist.data.astype(np.float32), 0)
# sigma = np.std(mnist.data.astype(np.float32), 0)
# mnist.data = (mnist.data.astype(np.float32) - mu)/(sigma+0.001)
# mnist_data = []
# for i in trange(10):
#     idx = mnist.target==str(i)
#     mnist_data.append(mnist.data[idx])


# print("\nNumb samples of each label:\n", [len(v) for v in mnist_data])
# users_lables = []
# print("idx",idx)
# # devide for label for each users:
# for user in trange(NUM_USERS):
#     for j in range(NUM_LABELS):  # 4 labels for each users
#         l = (user + j) % 10
#         users_lables.append(l)
# unique, counts = np.unique(users_lables, return_counts=True)
# print("--------------")
# print(unique, counts)

# def ram_dom_gen(total, size):
#     print(total)
#     nums = []
#     temp = []
#     for i in range(size - 1):
#         val = np.random.randint(total//(size + 1), total//2)
#         temp.append(val)
#         total -= val
#     temp.append(total)
#     print(temp)
#     return temp
# number_sample = []
# for total_value, count in zip(mnist_data, counts):
#     temp = ram_dom_gen(len(total_value), count)
#     number_sample.append(temp)
# print("--------------")
# print(number_sample)

# i = 0
# number_samples = []
# for i in range(len(number_sample[0])):
#     for sample in number_sample:
#         print(sample)
#         number_samples.append(sample[i])

# print("--------------")
# print(number_samples)


transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False,download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=len(trainset.data),shuffle=False)
testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset.data),shuffle=False)

for _, train_data in enumerate(trainloader,0):
    trainset.data, trainset.targets = train_data
for _, train_data in enumerate(testloader,0):
    testset.data, testset.targets = train_data

random.seed(1)
np.random.seed(1)
NUM_USERS = 50 # should be muitiple of 10
NUM_LABELS = 2

# numran1 = random.randint(10, 50)
# numran2 = random.randint(1, 10)
# num_samples = (num_samples) * numran2 + numran1 #+ 100

# Setup directory for train/test data
train_path = './data/Cifar10/data/train/cifar_niid_train.json'
test_path = './data/Cifar10/data/test/cifar_niid_test.json'
dir_path = os.path.dirname(train_path)
if not os.path.exists(dir_path):
    os.makedirs(dir_path)
dir_path = os.path.dirname(test_path)
if not os.path.exists(dir_path):
    os.makedirs(dir_path)

cifa_data_image = []
cifa_data_label = []

cifa_data_image.extend(trainset.data.cpu().detach().numpy())
cifa_data_image.extend(testset.data.cpu().detach().numpy())
cifa_data_label.extend(trainset.targets.cpu().detach().numpy())
cifa_data_label.extend(testset.targets.cpu().detach().numpy())
cifa_data_image = np.array(cifa_data_image)
cifa_data_label = np.array(cifa_data_label)

cifa_data = []
for i in trange(10):
    idx = cifa_data_label==i
    cifa_data.append(cifa_data_image[idx])


print("\nNumber of samples of each label:\n", [len(v) for v in cifa_data])
users_lables = []

# ***********************************************************************
print("idx",idx)
# devide for label for each users:
for user in trange(NUM_USERS):
    for j in range(NUM_LABELS):  # 4 labels for each users
        l = (user + j) % 10
        users_lables.append(l)
unique, counts = np.unique(users_lables, return_counts=True)
print("--------------")
print(unique, counts)

def ram_dom_gen(total, size):
    print(total)
    nums = []
    temp = []
    for i in range(size - 1):
        val = np.random.randint(total//(size + 1), total//2)
        temp.append(val)
        total -= val
    temp.append(total)
    print(temp)
    return temp
number_sample = []
for total_value, count in zip(cifa_data, counts):
    temp = ram_dom_gen(len(total_value), count)
    number_sample.append(temp)
print("--------------")
print(number_sample)

i = 0
number_samples = []
for i in range(len(number_sample[0])):
    for sample in number_sample:
        print(sample)
        number_samples.append(sample[i])

print("--------------")
print(number_samples)

#***********************************************************************



###### CREATE USER DATA SPLIT #######
# Assign 100 samples to each user
X = [[] for _ in range(NUM_USERS)]
y = [[] for _ in range(NUM_USERS)]
count = 0
for user in trange(NUM_USERS):
    for j in range(NUM_LABELS):  # 4 labels for each users
        l = (user + j) % 10
        print("value of L",l)
        print("value of count",count)
        num_samples =  number_samples[count] # num sample
        count = count + 1
        if idx[l] + num_samples < len(cifa_data[l]):
            X[user] += cifa_data[l][idx[l]:num_samples].tolist()
            y[user] += (l*np.ones(num_samples)).tolist()
            idx[l] += num_samples
            print("check len os user:", user, j,"len data", len(X[user]), num_samples)

print("IDX2:", idx) # counting samples for each labels

# Create data structure
train_data = {'users': [], 'user_data':{}, 'num_samples':[]}
test_data = {'users': [], 'user_data':{}, 'num_samples':[]}

# Setup 5 users
# for i in trange(5, ncols=120):
for i in range(NUM_USERS):
    uname = 'f_{0:05d}'.format(i)
    
    combined = list(zip(X[i], y[i]))
    random.shuffle(combined)
    X[i][:], y[i][:] = zip(*combined)
    num_samples = len(X[i])
    train_len = int(0.75*num_samples)
    test_len = num_samples - train_len
    
    train_data['users'].append(uname) 
    train_data['user_data'][uname] = {'x': X[i][:train_len], 'y': y[i][:train_len]}
    train_data['num_samples'].append(train_len)
    test_data['users'].append(uname)
    test_data['user_data'][uname] = {'x': X[i][train_len:], 'y': y[i][train_len:]}
    test_data['num_samples'].append(test_len)

print("Num_samples:", train_data['num_samples'])
print("Total_samples:",sum(train_data['num_samples'] + test_data['num_samples']))
    
with open(train_path,'w') as outfile:
    json.dump(train_data, outfile)
with open(test_path, 'w') as outfile:
    json.dump(test_data, outfile)

print("Finish Generating Samples")
