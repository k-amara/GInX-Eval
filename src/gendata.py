import torch
from torch import default_generator, randperm
from torch.utils.data import Subset
from torch_geometric.data import DataLoader
from dataset import (
    MoleculeDataset,
    SynGraphDataset,
    NCRealGraphDataset,
    Benzene,
    Mutag,
    MNIST75sp_Binary,
)
from torch import default_generator
from utils.parser_utils import arg_parse, get_graph_size_args


def get_dataset(dataset_root, **kwargs):
    dataset_name = kwargs.get("dataset_name")
    print(f"Loading {dataset_name} dataset...")
    if dataset_name.lower() in list(MoleculeDataset.names.keys()):
        return MoleculeDataset(root=dataset_root, name=dataset_name)
    elif dataset_name.lower() == "mutag":
        return Mutag(root=dataset_root, name=dataset_name)
    elif dataset_name.lower() == "benzene":
        return Benzene(root=dataset_root, name=dataset_name)
    elif dataset_name.lower() in list(NCRealGraphDataset.names.keys()):
        dataset = NCRealGraphDataset(
            root=dataset_root, name=dataset_name, dataset_params=kwargs
        )
        dataset.process()
        return dataset
    elif dataset_name.lower() == "mnist_bin":
        return MNIST75sp_Binary(root=dataset_root, name=dataset_name)
    elif dataset_name.lower() in list(SynGraphDataset.names.keys()):
        dataset = SynGraphDataset(
            root=dataset_root,
            name=dataset_name,
            transform=None,
            pre_transform=None,
            **kwargs,
        )
        # dataset.process()
        return dataset
    else:
        raise ValueError(f"{dataset_name} is not defined.")


def get_dataloader(
    dataset, batch_size, random_split_flag=True, data_split_ratio=None, seed=2
):
    """
    Args:
        dataset:
        batch_size: int
        random_split_flag: bool
        data_split_ratio: list, training, validation and testing ratio
        seed: random seed to split the dataset randomly
    Returns:
        a dictionary of training, validation, and testing dataLoader
    """

    if not random_split_flag and hasattr(dataset, "supplement"):
        assert "split_indices" in dataset.supplement.keys(), "split idx"
        split_indices = dataset.supplement["split_indices"]
        train_indices = torch.where(split_indices == 0)[0].numpy().tolist()
        dev_indices = torch.where(split_indices == 1)[0].numpy().tolist()
        test_indices = torch.where(split_indices == 2)[0].numpy().tolist()

        train = Subset(dataset, train_indices)
        eval = Subset(dataset, dev_indices)
        test = Subset(dataset, test_indices)
    else:
        num_train = int(data_split_ratio[0] * len(dataset))
        num_eval = int(data_split_ratio[1] * len(dataset))
        num_test = len(dataset) - num_train - num_eval

        from functools import partial

        lengths = [num_train, num_eval, num_test]
        indices = randperm(
            sum(lengths), generator=default_generator.manual_seed(seed)
        ).tolist()
        train_indices = indices[:num_train]
        dev_indices = indices[num_train : num_train + num_eval]
        test_indices = indices[num_train + num_eval :]

        # train, eval, test = random_split(
        # dataset,
        # lengths=[num_train, num_eval, num_test],
        # generator=default_generator.manual_seed(seed),
        # )

    train = Subset(dataset, train_indices)
    eval = Subset(dataset, dev_indices)
    test = Subset(dataset, test_indices)

    train_dataset = dataset[train_indices]
    eval_dataset = dataset[dev_indices]
    test_dataset = dataset[test_indices]

    # train.data, train.slices = train.collate([data for data in train])
    # eval.data, eval.slices = eval.collate([data for data in eval])
    # test.data, test.slices = test.collate([data for data in test])

    dataloader = dict()
    dataloader["train"] = DataLoader(train, batch_size=batch_size, shuffle=True)
    dataloader["eval"] = DataLoader(eval, batch_size=batch_size, shuffle=False)
    dataloader["test"] = DataLoader(test, batch_size=batch_size, shuffle=False)

    return dataloader, train_dataset, eval_dataset, test_dataset


if __name__ == "__main__":
    args = arg_parse()
    args = get_graph_size_args(args)
    data_params = {
        "num_shapes": args.num_shapes,
        "width_basis": args.width_basis,
        "input_dim": args.input_dim,
        "seed": args.seed,
        "test_ratio": args.test_ratio,
        "val_ratio": args.val_ratio,
    }
    dataset = get_dataset(args.data_save_dir, "ba_house", **data_params)
    # dataset = get_dataset(args.data_save_dir, "cora")
    print(dataset)
    print(dataset.data)
