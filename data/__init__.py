import importlib
import torch.utils.data
#from util.distributed import master_only_print as print
from torch.utils.data import Dataset
import numpy as np
import glob

def find_dataset_using_name(dataset_name):
    # Given the option --dataset [datasetname],
    dataset_filename = dataset_name
    module, target = dataset_name.split('::')
    datasetlib = importlib.import_module(module)
    # In the file, the class called`` DatasetNameDataset() will
    # be instantiated. It has to be a subclass of BaseDataset,
    # and it is case-insensitive.
    dataset = None
    # target_dataset_name = 'Dataset'
    for name, cls in datasetlib.__dict__.items():
        if name == target:
            dataset = cls
            
    if dataset is None:
        raise ValueError("In %s.py, there should be a class "
                         "with class name that matches %s in lowercase." %
                         (dataset_filename, target))

    return dataset


def get_option_setter(dataset_name):    
    dataset_class = find_dataset_using_name(dataset_name)
    return dataset_class.modify_commandline_options


def create_dataloader(opt, distributed, labels_required, is_inference):
    dataset = find_dataset_using_name(opt.type)
    instance = dataset(opt, is_inference, labels_required)
    phase = 'val' if is_inference else 'training'
    batch_size = opt.val.batch_size if is_inference else opt.train.batch_size
    print("%s dataset [%s] of size %d was created" %
          (phase, opt.type, len(instance)))
    
    dataloader = torch.utils.data.DataLoader(
        instance,
        batch_size=batch_size,
        sampler=data_sampler(instance, shuffle=not is_inference, distributed=distributed),
        drop_last=not is_inference,
        num_workers=getattr(opt, 'num_workers', 0),
    )          

    return dataloader


def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return torch.utils.data.distributed.DistributedSampler(dataset, shuffle=shuffle)
    if shuffle:
        return torch.utils.data.RandomSampler(dataset)
    else:
        return torch.utils.data.SequentialSampler(dataset)


def get_dataloader(opt, distributed, is_inference):
    dataset = create_dataloader(opt, distributed, is_inference)
    return dataset


def get_train_val_dataloader(opt, labels_required=False, distributed = False):


    val_dataset = create_dataloader(opt, distributed, labels_required = labels_required, is_inference=True,)
    train_dataset = create_dataloader(opt, distributed, labels_required = labels_required, is_inference=False)
        
    return val_dataset, train_dataset

def pad_images(img_tensor, pad_value):

    b,c,h,w = img_tensor.size()

    pad = torch.ones((b,c,h,(h-w)//2)).cuda()*pad_value

    return torch.cat([pad, img_tensor, pad], 3)








    





