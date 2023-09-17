# %%
import numpy as np 
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import torchvision.transforms as transforms

CHKP_PATH = r"C:\Users\TEMMMAR\Desktop\Hifi_local\Chekpoint\hific-high.pt"

device = torch.device("cuda")
batch_size = 8
workers = 8 

TEST_LABEL_PATH = r"C:\Users\TEMMMAR\Desktop\PFE\classifier\test_labels.npy"
LABEL_PATH = r"C:\Users\TEMMMAR\Desktop\PFE\classifier\all_labels.npy"  # .np file containing the 50000 labels 




class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]

        if self.transform is not None:
            x = self.transform(x)

        return x, y

    def __len__(self):
        return len(self.data)
    


def Data_load(DATA_PATH, use_data_aug = False):
    Data_np = np.load(DATA_PATH)
    Data_np = Data_np/Data_np.max()
    Data_np= Data_np.transpose(0,3,1,2)
    print(Data_np.shape)
    Data_tensor = torch.Tensor(Data_np)
    # B = Data_tensor.shape[0]
    x_dims = tuple(Data_tensor.size())
    
    # Create an instance of the custom dataset with data augmentation
    if use_data_aug:
        # Define transformations for data augmentation
        transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(64, padding=4),
        # transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])  # normalize the images
        ])
        my_dataset = CustomDataset(Data_tensor, get_labels(LABEL_PATH), transform=transform)
    else:
         my_dataset = TensorDataset(Data_tensor,get_labels(LABEL_PATH)) 


    return  DataLoader(my_dataset, batch_size=batch_size,num_workers=workers,shuffle=True), x_dims


def test_load(DATA_PATH, use_data_aug = False):
    Data_np = np.load(DATA_PATH)
    Data_np = Data_np/Data_np.max()
    Data_np= Data_np.transpose(0,3,1,2)
    print(Data_np.shape)
    Data_tensor = torch.Tensor(Data_np)

    # Create an instance of the custom dataset with data augmentation
    if use_data_aug:
        # Define transformations for data augmentation
        transform = transforms.Compose([
        # transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])  # normalize the images
        ])
        my_dataset = CustomDataset(Data_tensor, get_labels(TEST_LABEL_PATH), transform=transform)
    else:
         my_dataset = TensorDataset(Data_tensor,get_labels(TEST_LABEL_PATH)) 

    return  DataLoader(my_dataset, batch_size=batch_size,num_workers=workers,shuffle=True) 


def get_labels(path): 
    label_np = np.load(path)
    Data_tensor = torch.tensor(label_np).long()
    Data_tensor = torch.reshape(Data_tensor, (-1,))# (50000,1))
    return Data_tensor