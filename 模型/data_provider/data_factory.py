from data_provider.data_stock import SDataset
from torch.utils.data import DataLoader


data_dict = {
    'Stock': SDataset,
}


def data_provider(args, flag):
    Data = SDataset
    drop_last = False
    batch_size = args.batch_size
    num_workers = getattr(args, 'num_workers', 0)
    
    data_set = Data(
        args=args,
        flag=flag,
    )
    loader_kwargs = {
        'batch_size': batch_size,
        'drop_last': drop_last,
        'num_workers': num_workers,
    }
    if num_workers > 0:
        loader_kwargs['persistent_workers'] = True
        loader_kwargs['pin_memory'] = True

    data_loader = DataLoader(
        data_set,
        **loader_kwargs,
    )

    return data_set, data_loader