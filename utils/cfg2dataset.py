from dl_lib.datasets.ready_datasets import get_ModelNet40, get_scanObjectNN



def get_cls_dataloader(cfg):

    dataset = cfg['dataset']['name']
    path    = cfg['dataset']['path']

    if dataset == 'modelnet':
        train_loader, valid_loader = get_ModelNet40(path, 'normalized')

    elif dataset == 'scanObjectNN':
        train_loader, valid_loader = get_scanObjectNN(path, 'easy')
    
    elif dataset == 'scanObjectNN-hard':
        train_loader, valid_loader = get_scanObjectNN(path, 'hard')
    
    else:
        raise NotImplementedError

    return train_loader, valid_loader