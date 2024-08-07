import os
import sys
import time

import torch
from autocuda import auto_cuda, auto_cuda_name
from termcolor import colored

from mini_pyabsa import __version__ as pyabsa_version
from mini_pyabsa.framework.flag_class.flag_template import DeviceTypeOption


def save_args(config, save_path):
    """
    Save arguments to a file.

    Args:
    - config: A Namespace object containing the arguments.
    - save_path: A string representing the path of the file to be saved.

    Returns:
    None
    """
    f = open(os.path.join(save_path), mode="w", encoding="utf8")
    for arg in config.args:
        if config.args_call_count[arg]:
            f.write("{}: {}\n".format(arg, config.args[arg]))
    f.close()


def print_args(config, logger=None):
    """
    Print the arguments to the console.

    Args:
    - config: A Namespace object containing the arguments.
    - logger: A logger object.

    Returns:
    None
    """
    args = [key for key in sorted(config.args.keys())]
    for arg in args:
        if arg != "dataset" and arg != "dataset_dict" and arg != "embedding_matrix":
            if logger:
                try:
                    logger.info(
                        "{0}:{1}\t-->\tCalling Count:{2}".format(
                            arg, config.args[arg], config.args_call_count[arg]
                        )
                    )
                except:
                    logger.info(
                        "{0}:{1}\t-->\tCalling Count:{2}".format(
                            arg, config.args[arg], 0
                        )
                    )
            else:
                try:
                    fprint(
                        "{0}:{1}\t-->\tCalling Count:{2}".format(
                            arg, config.args[arg], config.args_call_count[arg]
                        )
                    )
                except:
                    fprint(
                        "{0}:{1}\t-->\tCalling Count:{2}".format(
                            arg, config.args[arg], 0
                        )
                    )



def set_device(config, auto_device):
    """
    Sets the device to be used for the PyTorch model.

    :param config: An instance of ConfigManager class that holds the configuration for the model.
    :param auto_device: Specifies the device to be used for the model. It can be either a string, a boolean, or None.
                        If it is a string, it can be either "cuda", "cuda:0", "cuda:1", or "cpu".
                        If it is a boolean and True, it automatically selects the available CUDA device.
                        If it is None, it uses the autocuda.
    :return: device: The device to be used for the PyTorch model.
             device_name: The name of the device.
    """
    device_name = "Unknown"
    if isinstance(auto_device, str) and auto_device == DeviceTypeOption.ALL_CUDA:
        device = "cuda"
    elif isinstance(auto_device, str):
        device = auto_device
    elif isinstance(auto_device, bool):
        device = auto_cuda() if auto_device else DeviceTypeOption.CPU
    else:
        device = auto_cuda()
        try:
            torch.device(device)
        except RuntimeError as e:
            print(
                colored("Device assignment error: {}, redirect to CPU".format(e), "red")
            )
            device = DeviceTypeOption.CPU
    if device != DeviceTypeOption.CPU:
        device_name = auto_cuda_name()
    config.device = device
    config.device_name = device_name
    fprint("Set Model Device: {}".format(device))
    fprint("Device Name: {}".format(device_name))
    return device, device_name


def fprint(*objects, sep=" ", end="\n", file=sys.stdout, flush=False):
    """
    Custom print function that adds a timestamp and the pyabsa version before the printed message.

    Args:
        *objects: Any number of objects to be printed
        sep (str, optional): Separator between objects. Defaults to " ".
        end (str, optional): Ending character after all objects are printed. Defaults to "\n".
        file (io.TextIOWrapper, optional): Text file to write printed output to. Defaults to sys.stdout.
        flush (bool, optional): Whether to flush output buffer after printing. Defaults to False.
    """
    print(
        time.strftime(
            "[%Y-%m-%d %H:%M:%S] ({})".format(pyabsa_version),
            time.localtime(time.time()),
        ),
        *objects,
        sep=sep,
        end=end,
        file=file,
        flush=flush
    )


def init_optimizer(optimizer):
    """
    Initialize the optimizer for the PyTorch model.

    Args:
        optimizer: str or PyTorch optimizer object.

    Returns:
        PyTorch optimizer object.

    Raises:
        KeyError: If the optimizer is unsupported.
    """
    optimizers = {
        "adadelta": torch.optim.Adadelta,  # default lr=1.0
        "adagrad": torch.optim.Adagrad,  # default lr=0.01
        "adam": torch.optim.Adam,  # default lr=0.001
        "adamax": torch.optim.Adamax,  # default lr=0.002
        "asgd": torch.optim.ASGD,  # default lr=0.01
        "rmsprop": torch.optim.RMSprop,  # default lr=0.01
        "sgd": torch.optim.SGD,
        "adamw": torch.optim.AdamW,
        torch.optim.Adadelta: torch.optim.Adadelta,  # default lr=1.0
        torch.optim.Adagrad: torch.optim.Adagrad,  # default lr=0.01
        torch.optim.Adam: torch.optim.Adam,  # default lr=0.001
        torch.optim.Adamax: torch.optim.Adamax,  # default lr=0.002
        torch.optim.ASGD: torch.optim.ASGD,  # default lr=0.01
        torch.optim.RMSprop: torch.optim.RMSprop,  # default lr=0.01
        torch.optim.SGD: torch.optim.SGD,
        torch.optim.AdamW: torch.optim.AdamW,
    }
    if optimizer in optimizers:
        return optimizers[optimizer]
    elif hasattr(torch.optim, optimizer.__name__):
        return optimizer
    else:
        raise KeyError(
            "Unsupported optimizer: {}. "
            "Please use string or the optimizer objects in torch.optim as your optimizer".format(
                optimizer
            )
        )
