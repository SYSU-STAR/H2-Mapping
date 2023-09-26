import argparse
from importlib import import_module


def get_dataset(args):
    Dataset = import_module("dataset." + args.dataset)
    return Dataset.DataLoader(**args.data_specs)


def get_decoder(args):
    Decoder = import_module("functions." + args.decoder)
    return Decoder.Decoder(**args.decoder_specs)


def get_property(args, name, default):
    if isinstance(args, dict):
        return args.get(name, default)
    elif isinstance(args, argparse.Namespace):
        if hasattr(args, name):
            return vars(args)[name]
        else:
            return default
    else:
        raise ValueError(f"unkown dict/namespace type: {type(args)}")
