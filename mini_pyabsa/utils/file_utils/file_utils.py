import json
import os
import pickle
import sys
import zipfile
from typing import Union, List

import numpy as np
import requests
import torch
import tqdm
from findfile import find_files, find_cwd_file
from termcolor import colored

from mini_pyabsa.utils.pyabsa_utils import save_args, fprint


def load_dataset_from_file(fname, config):
    """
    Loads a dataset from one or multiple files.

    Args:
        fname (str or List[str]): The name of the file(s) containing the dataset.
        config (dict): The configuration dictionary containing the logger (optional) and the maximum number of data to load (optional).

    Returns:
        A list of strings containing the loaded dataset.

    Raises:
        ValueError: If an empty line is found in the dataset.

    """
    logger = config.get("logger", None)
    lines = []
    if isinstance(fname, str):
        fname = [fname]

    for f in fname:
        if logger:
            logger.info("Load dataset from {}".format(f))
        else:
            fprint("Load dataset from {}".format(f))
        fin = open(f, "r", encoding="utf-8")
        _lines_ = fin.readlines()
        for i, line in enumerate(_lines_):
            if not line.strip():
                raise ValueError(
                    "empty line: #{} in {}, previous line: {}".format(
                        i, f, _lines_[i - 1]
                    )
                )
            lines.append(line.strip())
        fin.close()
    lines = lines[: config.get("data_num", None)]
    return lines


def prepare_glove840_embedding(glove_path, embedding_dim, config):
    """
    Check if the provided GloVe embedding exists, if not, search for a similar file in the current directory, or download
    the 840B GloVe embedding. If none of the above exists, raise an error.
    :param glove_path: str, path to the GloVe embedding
    :param embedding_dim: int, the dimension of the embedding
    :param config: dict, configuration dictionary
    :return: str, the path to the GloVe embedding
    """
    if config.get("glove_or_word2vec_path", None):
        glove_path = config.glove_or_word2vec_path
        return glove_path

    logger = config.logger
    if os.path.exists(glove_path) and os.path.isfile(glove_path):
        return glove_path
    else:
        embedding_files = []
        dir_path = os.getenv("$HOME") if os.getenv("$HOME") else os.getcwd()

        if find_files(
            dir_path,
            ["glove", "B", "d", ".txt", str(embedding_dim)],
            exclude_key=".zip",
        ):
            embedding_files += find_files(
                dir_path, ["glove", "B", ".txt", str(embedding_dim)], exclude_key=".zip"
            )
        elif find_files(dir_path, ["word2vec", "d", ".txt"], exclude_key=".zip"):
            embedding_files += find_files(
                dir_path,
                ["word2vec", "d", ".txt", str(embedding_dim)],
                exclude_key=".zip",
            )
        else:
            embedding_files += find_files(
                dir_path, ["d", ".txt", str(embedding_dim)], exclude_key=".zip"
            )

        if embedding_files:
            logger.info(
                "Find embedding file: {}, use: {}".format(
                    embedding_files, embedding_files[0]
                )
            )
            return embedding_files[0]

        else:
            if config.embed_dim != 300:
                raise ValueError(
                    "Please provide embedding file for embedding dim: {} in current wording dir ".format(
                        config.embed_dim
                    )
                )
            zip_glove_path = os.path.join(
                os.path.dirname(glove_path), "glove.840B.300d.zip"
            )
            logger.info(
                "No GloVe embedding found at {},"
                " downloading glove.840B.300d.txt (2GB will be downloaded / 5.5GB after unzip)".format(
                    glove_path
                )
            )
            try:
                response = requests.get(
                    "https://huggingface.co/spaces/yangheng/PyABSA-ATEPC/resolve/main/open-access/glove.840B.300d.zip",
                    stream=True,
                )
                with open(zip_glove_path, "wb") as f:
                    for chunk in tqdm.tqdm(
                        response.iter_content(chunk_size=1024 * 1024),
                        unit="MB",
                        total=int(response.headers["content-length"]) // 1024 // 1024,
                        desc=colored("Downloading GloVe-840B embedding", "yellow"),
                    ):
                        f.write(chunk)
            except Exception as e:
                raise ValueError(
                    "Download failed, please download glove.840B.300d.zip from "
                    "https://nlp.stanford.edu/projects/glove/, unzip it and put it in {}.".format(
                        glove_path
                    )
                )

        if find_cwd_file("glove.840B.300d.zip"):
            logger.info("unzip glove.840B.300d.zip")
            with zipfile.ZipFile(find_cwd_file("glove.840B.300d.zip"), "r") as z:
                z.extractall()
            logger.info("Zip file extraction Done.")

        return prepare_glove840_embedding(glove_path, embedding_dim, config)


def unzip_checkpoint(zip_path):
    """
    Unzip a checkpoint file in zip format.

    Args:
        zip_path (str): path to the zip file.

    Returns:
        str: path to the unzipped checkpoint directory.

    """
    try:
        # Inform the user that the checkpoint file is being unzipped
        print("Find zipped checkpoint: {}, unzipping".format(zip_path))
        sys.stdout.flush()

        # Create a directory with the same name as the zip file to store the unzipped checkpoint files
        if not os.path.exists(zip_path):
            os.makedirs(zip_path.replace(".zip", ""))

        # Extract the contents of the zip file to the created directory
        z = zipfile.ZipFile(zip_path, "r")
        z.extractall(os.path.dirname(zip_path))

        # Inform the user that the unzipping is done
        print("Done.")
    except zipfile.BadZipfile:
        # If the zip file is corrupted, inform the user that the unzipping has failed
        print("{}: Unzip failed".format(zip_path))

    # Return the path to the unzipped checkpoint directory
    return zip_path.replace(".zip", "")


def save_model(config, model, tokenizer, save_path, **kwargs):
    """
    Save a trained model, configuration, and tokenizer to the specified path.

    Args:
        config (Config): Configuration for the model.
        model (nn.Module): The trained model.
        tokenizer: Tokenizer used by the model.
        save_path (str): The path where to save the model, config, and tokenizer.
        **kwargs: Additional keyword arguments.
    """
    if (
        hasattr(model, "module")
        or hasattr(model, "core")
        or hasattr(model, "_orig_mod")
    ):
        model_to_save = model.module
    else:
        model_to_save = model
    # Check the specified save mode.
    if config.save_mode == 1 or config.save_mode == 2:
        # Create save_path directory if it doesn't exist.
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # Save the configuration and tokenizer to the save_path directory.
        f_config = open(save_path + config.model_name + ".config", mode="wb")
        f_tokenizer = open(save_path + config.model_name + ".tokenizer", mode="wb")
        pickle.dump(config, f_config)
        pickle.dump(tokenizer, f_tokenizer)
        f_config.close()
        f_tokenizer.close()
        # Save the arguments used to create the configuration.
        save_args(config, save_path + config.model_name + ".args.txt")
        # Save the model state dict or the entire model depending on the save mode.
        if config.save_mode == 1:
            torch.save(
                model_to_save.state_dict(),
                save_path + config.model_name + ".state_dict",
            )
        elif config.save_mode == 2:
            torch.save(model.cpu(), save_path + config.model_name + ".model")

    elif config.save_mode == 3:
        # Save the fine-tuned BERT model.
        model_output_dir = save_path + "fine-tuned-pretrained-model"
        if not os.path.exists(model_output_dir):
            os.makedirs(model_output_dir)
        output_model_file = os.path.join(model_output_dir, "pytorch_model.bin")
        output_config_file = os.path.join(model_output_dir, "config.json")

        if hasattr(model_to_save, "bert4global"):
            model_to_save = model_to_save.bert4global
        elif hasattr(model_to_save, "bert"):
            model_to_save = model_to_save.bert

        torch.save(model_to_save.state_dict(), output_model_file)
        model_to_save.config.to_json_file(output_config_file)
        if hasattr(tokenizer, "tokenizer"):
            tokenizer.tokenizer.save_pretrained(model_output_dir)
        else:
            tokenizer.save_pretrained(model_output_dir)

    else:
        raise ValueError("Invalid save_mode: {}".format(config.save_mode))
    model.to(config.device)