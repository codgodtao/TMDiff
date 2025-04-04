'''create dataset and dataloader'''
import logging
from re import split
import torch.utils.data


def create_dataloader(dataset, dataset_opt, phase):
    '''create dataloader '''
    if 'train' in phase:
        # sampler = torch.utils.data.DistributedSampler(dataset)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=dataset_opt['batch_size'],
            shuffle=dataset_opt['use_shuffle'],
            num_workers=dataset_opt['num_workers'],
            pin_memory=True)
    else:
        return torch.utils.data.DataLoader(
            dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)


def create_dataset(dataset_opt, phase):
    '''create dataset'''
    from data.LRHR_dataset import LRHRDataset as D
    dataset = D(dataroot=dataset_opt['dataroot'],
                data_len=dataset_opt['data_len'],
                phase=phase)

    return dataset


def create_dataset2(dataset_opt, phase):
    '''create dataset'''
    from data.LRHR_dataset import LRHRDataset as D
    dataset = D(dataroot=dataset_opt['dataroot'],
                data_len=dataset_opt['data_len'],
                phase=phase)
    return dataset
