
import numpy as np
import os
import torch
import torchvision
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, random_split
from torchvision import transforms
import PIL.Image as Image
from sklearn.datasets import load_boston
np.random.seed(42)
torch.manual_seed(42)

#TODO: Add func for att imbalance
from torch.utils.data import Dataset
class custom_subset(Dataset):
    r"""
    Subset of a dataset at specified indices.

    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
        labels(sequence) : targets as required for the indices. will be the same length as indices
    """
    def __init__(self, dataset, indices, labels):
        self.dataset = torch.utils.data.Subset(dataset, indices)
        self.targets = labels.type(torch.long)
    def __getitem__(self, idx):
        image = self.dataset[idx][0]
        target = self.targets[idx]
        return (image, target)

    def __len__(self):
        return len(self.targets)

class DataHandler_CIFAR10(Dataset):
    """
    Data Handler to load CIFAR10 dataset.
    This class extends :class:`torch.utils.data.Dataset` to handle 
    loading data even without labels

    Parameters
    ----------
    X: numpy array
        Data to be loaded   
    y: numpy array, optional
        Labels to be loaded (default: None)
    select: bool
        True if loading data without labels, False otherwise
    """

    def __init__(self, X, Y=None, select=True, use_test_transform = False):
        """
        Constructor
        """
        self.select = select
        if(use_test_transform):
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        else:
            transform = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        if not self.select:
            self.X = X
            self.targets = Y
            self.transform = transform
        else:
            self.X = X
            self.transform = transform

    def __getitem__(self, index):
        if not self.select:
            x, y = self.X[index], self.targets[index]
            x = Image.fromarray(x)
            x = self.transform(x)
            return (x, y)

        else:
            x = self.X[index]
            x = Image.fromarray(x)
            x = self.transform(x)
            return x

    def __len__(self):
        return len(self.X)

def create_random_train(fullset, split_cfg, num_cls, isnumpy, augVal):
    np.random.seed(42)
    full_idx = list(range(len(fullset)))
    train_idx = list(np.random.choice(np.array(full_idx), size=split_cfg['train_size'], replace=False))
    lake_idx = list(set(full_idx) - set(train_idx))
    val_idx = list(np.random.choice(np.array(lake_idx), size=split_cfg['val_size'], replace=False))
    lake_idx = list(set(lake_idx) - set(val_idx))
    train_set = custom_subset(fullset, train_idx, torch.Tensor(fullset.targets)[train_idx])
    val_set = custom_subset(fullset, val_idx, torch.Tensor(fullset.targets)[val_idx])
    lake_set = custom_subset(fullset, lake_idx, torch.Tensor(fullset.targets)[lake_idx])
    return train_set, val_set, lake_set

def create_uniform_train(fullset, split_cfg, num_cls, isnumpy, augVal):
    np.random.seed(42)
    train_idx = []
    val_idx = []
    lake_idx = []
    for i in range(num_cls): #all_classes
        full_idx_class = list(torch.where(torch.Tensor(fullset.targets) == i)[0].cpu().numpy())
        class_train_idx = list(np.random.choice(np.array(full_idx_class), size=split_cfg['train_size']//num_cls, replace=False))
        remain_idx = list(set(full_idx_class) - set(class_train_idx))
        train_idx += class_train_idx
        lake_idx += remain_idx
    val_idx =  list(np.random.choice(np.array(lake_idx), size=split_cfg['val_size'], replace=False))
    lake_idx = list(set(lake_idx) - set(val_idx))
    train_set = custom_subset(fullset, train_idx, torch.Tensor(fullset.targets)[train_idx])
    val_set = custom_subset(fullset, val_idx, torch.Tensor(fullset.targets)[val_idx])
    lake_set = custom_subset(fullset, lake_idx, torch.Tensor(fullset.targets)[lake_idx])
    return train_set, val_set, lake_set

def create_class_imb(fullset, split_cfg, num_cls, isnumpy, augVal):
    np.random.seed(42)
    train_idx = []
    val_idx = []
    lake_idx = []
    selected_classes = np.random.choice(np.arange(num_cls), size=split_cfg['num_cls_imbalance'], replace=False) #classes to imbalance
    for i in range(num_cls): #all_classes
        full_idx_class = list(torch.where(torch.Tensor(fullset.targets) == i)[0].cpu().numpy())
        if(i in selected_classes):
            class_train_idx = list(np.random.choice(np.array(full_idx_class), size=split_cfg['per_imbclass_train'], replace=False))
            remain_idx = list(set(full_idx_class) - set(class_train_idx))
            class_val_idx = list(np.random.choice(np.array(remain_idx), size=split_cfg['per_imbclass_val'], replace=False))
            remain_idx = list(set(remain_idx) - set(class_val_idx))
            class_lake_idx = list(np.random.choice(np.array(remain_idx), size=split_cfg['per_imbclass_lake'], replace=False))
        else:
            class_train_idx = list(np.random.choice(np.array(full_idx_class), size=split_cfg['per_class_train'], replace=False))
            remain_idx = list(set(full_idx_class) - set(class_train_idx))
            class_val_idx = list(np.random.choice(np.array(remain_idx), size=split_cfg['per_class_val'], replace=False))
            remain_idx = list(set(remain_idx) - set(class_val_idx))
            class_lake_idx = list(np.random.choice(np.array(remain_idx), size=split_cfg['per_class_lake'], replace=False))
    
        train_idx += class_train_idx
        if(augVal and (i in selected_classes)): #augment with samples only from the imbalanced classes
            train_idx += class_val_idx
        val_idx += class_val_idx
        lake_idx += class_lake_idx

    train_set = custom_subset(fullset, train_idx, torch.Tensor(fullset.targets)[train_idx])
    val_set = custom_subset(fullset, val_idx, torch.Tensor(fullset.targets)[val_idx])
    lake_set = custom_subset(fullset, lake_idx, torch.Tensor(fullset.targets)[lake_idx])
    if(isnumpy):        
        X = fullset.data
        y = torch.from_numpy(np.array(fullset.targets))
        X_tr = X[train_idx]
        y_tr = y[train_idx]
        X_val = X[val_idx]
        y_val = y[val_idx]
        X_unlabeled = X[lake_idx]
        y_unlabeled = y[lake_idx]
        return X_tr, y_tr, X_val, y_val, X_unlabeled, y_unlabeled, train_set, val_set, lake_set, selected_classes
    else:
        return train_set, val_set, lake_set, selected_classes

#TODO: Add attimb, duplicates, weak aug, out-of-dist settings
def load_dataset_custom(datadir, dset_name, feature, split_cfg, isnumpy=False, augVal=False, transform=False):
    if(not(os.path.exists(datadir))):
        os.mkdir(datadir)

    if(dset_name=="cifar10"):
        num_cls=10
        if(transform):
            cifar_test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
            cifar_transform = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        else:
            cifar_test_transform = transforms.Compose([transforms.ToTensor()])
            cifar_transform = cifar_test_transform
        fullset = torchvision.datasets.CIFAR10(root=datadir, train=True, download=True, transform=cifar_transform)
        test_set = torchvision.datasets.CIFAR10(root=datadir, train=False, download=True, transform=cifar_test_transform)
        if(feature=="uniform"):
            train_set, val_set, lake_set = create_uniform_train(fullset, split_cfg, num_cls, isnumpy, augVal)
            return train_set, val_set, test_set, lake_set, num_cls
        if(feature=="random"):
            train_set, val_set, lake_set = create_random_train(fullset, split_cfg, num_cls, isnumpy, augVal)
            return train_set, val_set, test_set, lake_set, num_cls
        if(feature=="classimb"):
            if(isnumpy):
                X_tr, y_tr, X_val, y_val, X_unlabeled, y_unlabeled, train_set, val_set, lake_set, imb_cls_idx = create_class_imb(fullset, split_cfg, num_cls, isnumpy, augVal)
                print("CIFAR-10 Custom dataset stats: Train size: ", len(train_set), "Val size: ", len(val_set), "Lake size: ", len(lake_set))
                return X_tr, y_tr, X_val, y_val, X_unlabeled, y_unlabeled, train_set, val_set, test_set, lake_set, imb_cls_idx, num_cls
            else:    
                train_set, val_set, lake_set, imb_cls_idx = create_class_imb(fullset, split_cfg, num_cls, isnumpy, augVal)
                print("CIFAR-10 Custom dataset stats: Train size: ", len(train_set), "Val size: ", len(val_set), "Lake size: ", len(lake_set))
                return train_set, val_set, test_set, lake_set, imb_cls_idx, num_cls
        if(feature=="vanilla"):
            X = fullset.data
            y = torch.from_numpy(np.array(fullset.targets))
            X_tr = X[:split_cfg['train_size']]
            y_tr = y[:split_cfg['train_size']]
            X_unlabeled = X[split_cfg['train_size']:len(X)-split_cfg['val_size']]
            y_unlabeled = y[split_cfg['train_size']:len(X)-split_cfg['val_size']]
            X_val = X[len(X)-split_cfg['val_size']:]
            y_val = y[len(X)-split_cfg['val_size']:]
            train_set = DataHandler_CIFAR10(X_tr, y_tr, False)
            lake_set = DataHandler_CIFAR10(X_unlabeled[:split_cfg['lake_size']], y_unlabeled[:split_cfg['lake_size']], False)
            val_set = DataHandler_CIFAR10(X_val, y_val, False)
            print("CIFAR-10 Custom dataset stats: Train size: ", len(train_set), "Val size: ", len(val_set), "Lake size: ", len(lake_set))
            if(isnumpy):
                return X_tr, y_tr, X_val, y_val, X_unlabeled[:split_cfg['lake_size']], y_unlabeled[:split_cfg['lake_size']], train_set, val_set, test_set, lake_set, num_cls
            else:
                return train_set, val_set, test_set, lake_set, num_cls
        if(feature=="duplicate"):
            num_rep=split_cfg['num_rep']
            X = fullset.data
            y = torch.from_numpy(np.array(fullset.targets))
            X_tr = X[:split_cfg['train_size']]
            y_tr = y[:split_cfg['train_size']]
            X_unlabeled = X[split_cfg['train_size']:len(X)-split_cfg['val_size']]
            y_unlabeled = y[split_cfg['train_size']:len(X)-split_cfg['val_size']]
            X_val = X[len(X)-split_cfg['val_size']:]
            y_val = y[len(X)-split_cfg['val_size']:]
            X_unlabeled_rep = np.repeat(X_unlabeled[:split_cfg['lake_subset_repeat_size']], num_rep, axis=0)
            y_unlabeled_rep = np.repeat(y_unlabeled[:split_cfg['lake_subset_repeat_size']], num_rep, axis=0)
            assert((X_unlabeled_rep[0]==X_unlabeled_rep[num_rep-1]).all())
            assert((y_unlabeled_rep[0]==y_unlabeled_rep[num_rep-1]).all())
            X_unlabeled_rep = np.concatenate((X_unlabeled_rep, X_unlabeled[split_cfg['lake_subset_repeat_size']:split_cfg['lake_size']]), axis=0)
            y_unlabeled_rep = torch.from_numpy(np.concatenate((y_unlabeled_rep, y_unlabeled[split_cfg['lake_subset_repeat_size']:split_cfg['lake_size']]), axis=0))
            train_set = DataHandler_CIFAR10(X_tr, y_tr, False)
            lake_set = DataHandler_CIFAR10(X_unlabeled_rep, y_unlabeled_rep, False)
            val_set = DataHandler_CIFAR10(X_val, y_val, False)
            print("CIFAR-10 Custom dataset stats: Train size: ", len(train_set), "Val size: ", len(val_set), "Lake size: ", len(lake_set))
            if(isnumpy):
                return X_tr, y_tr, X_unlabeled_rep, y_unlabeled_rep, train_set, val_set, test_set, lake_set, num_cls
            else:
                return train_set, val_set, test_set, lake_set, num_cls
        

    if(dset_name=="mnist"):
        num_cls=10
        mnist_transform = transforms.Compose([
            torchvision.transforms.ToTensor(),
            transforms.Resize((32, 32)),
            torchvision.transforms.Normalize([0.5], [0.5])
        ])
        fullset = torchvision.datasets.MNIST(root=datadir, train=True, download=True, transform=mnist_transform)
        test_set = torchvision.datasets.MNIST(root=datadir, train=False, download=True, transform=mnist_transform)
        if(feature=="classimb"):
            train_set, val_set, lake_set, imb_cls_idx = create_class_imb(fullset, split_cfg, num_cls)
        print("MNIST Custom dataset stats: Train size: ", len(train_set), "Val size: ", len(val_set), "Lake size: ", len(lake_set))

        return train_set, val_set, test_set, lake_set, imb_cls_idx, num_cls

    if(dset_name=="cifar100"):
        num_cls=100
        cifar100_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        fullset = torchvision.datasets.CIFAR100(root=datadir, train=True, download=True, transform=cifar100_transform)
        test_set = torchvision.datasets.CIFAR100(root=datadir, train=False, download=True, transform=cifar100_transform)
        if(feature=="classimb"):
            train_set, val_set, lake_set, imb_cls_idx = create_class_imb(fullset, split_cfg, num_cls)
        print("CIFAR-100 Custom dataset stats: Train size: ", len(train_set), "Val size: ", len(val_set), "Lake size: ", len(lake_set))

        return train_set, val_set, test_set, lake_set, imb_cls_idx, num_cls

