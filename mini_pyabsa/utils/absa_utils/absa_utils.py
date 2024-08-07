import copy
import os
import re
import string
import findfile
from mini_pyabsa.utils.data_utils.dataset_item import DatasetItem
from mini_pyabsa.utils.pyabsa_utils import fprint
from mini_pyabsa.framework.flag_class.flag_template import TaskCodeOption


def convert_apc_set_to_atepc_set(path, use_tokenizer=False):
    """
    Converts APC dataset to ATEPC dataset.
    :param path: path to the dataset
    :param use_tokenizer: whether to use a tokenizer
    """
    fprint(
        'To ensure your conversion is successful, make sure the dataset name contain "apc" and "dataset" string '
    )

    if isinstance(path, DatasetItem):
        path = path.dataset_name
    if os.path.isfile(path):
        files = [path]
    elif os.path.exists(path):
        files = findfile.find_files(
            path,
            ["dataset", TaskCodeOption.Aspect_Polarity_Classification],
            exclude_key=[".inference", "readme"],
        )
    else:
        files = findfile.find_cwd_files(
            [path, "dataset", TaskCodeOption.Aspect_Polarity_Classification],
            exclude_key=[".inference", "readme"],
        )

    fprint("Find datasets files at {}:".format(path))
    for target_file in files:
        if not target_file.endswith(".atepc"):
            try:
                convert_atepc(target_file, use_tokenizer)
            except Exception as e:
                fprint("failed to process :{}, Exception: {}".format(target_file, e))
        else:
            fprint("Ignore ", target_file)
    fprint("finished")

def simple_split_text(text):
    # text = ' '.join(tokenizer.tokenize(text)[1:])
    # return text
    text = text.strip()

    Chinese_punctuation = "＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､　、〃〈〉《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏﹑﹔·！？｡。"
    punctuation = string.punctuation + Chinese_punctuation
    for p in punctuation:
        text = text.replace("{}".format(p), " {} ".format(p))
    # text = ' '.join(re.compile(r'\w+|[{}]'.format(re.escape(punctuation))).findall(text)).replace('$ T $', '$T$')

    # for non-latin Languages
    non_latin_unicode = [
        "\u4e00-\u9fa5",  # Chinese
        "\u0800-\u4e00",  # Japanese
        "\uac00-\ud7a3",  # Korean
        "\u0e00-\u0e7f",  # Thai
        "\u1000-\u109F",  # Myanmar
    ]
    # latin_lan = ([re.match(lan, text) for lan in non_latin_unicode])
    latin_lan = [re.findall("[{}]".format(lan), text) for lan in non_latin_unicode]
    if not any(latin_lan):
        return text.split()

    s = text
    word_list = []
    while len(s) > 0:
        match_ch = re.match("[{}]".format("".join(non_latin_unicode)), s)
        if match_ch:
            word = s[0:1]
        else:
            match_en = re.match(r"[a-zA-Z\d]+", s)
            if match_en:
                word = match_en.group(0)
            else:
                word = s[0:1] 
        if word:
            word_list.append(word)
        s = s.replace(word, "", 1).strip(" ")
    return word_list

def assemble_aspects(fname, use_tokenizer=False):
    """
    Preprocesses the input file, groups sentences with similar aspects, and generates samples with the corresponding aspect labels and polarities.

    :param fname: The filename to be preprocessed
    :type fname: str
    :param use_tokenizer: Whether to use a tokenizer, defaults to False
    :type use_tokenizer: bool, optional
    :return: A list of samples
    :rtype: list
    """
    # Import tokenizer from transformers library if `use_tokenizer` is True
    if use_tokenizer:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")

    # Open and read the input file
    fin = open(fname, "r", encoding="utf-8", newline="\n", errors="ignore")
    lines = fin.readlines()
    fin.close()

    # Raise an error if an empty line is found
    for i, line in enumerate(lines):
        if not line.strip():
            raise ValueError(
                "empty line: #{}, previous line: {}".format(i, lines[i - 1])
            )

    # Preprocess the data by replacing tokens and splitting the text into tokens
    for i in range(len(lines)):
        if i % 3 == 0 or i % 3 == 1:
            if use_tokenizer:
                lines[i] = (
                    " ".join(tokenizer.tokenize(lines[i].strip()))
                    .replace("$ t $", "$T$")
                    .replace("$ T $", "$T$")
                )
            else:
                lines[i] = (
                    " ".join(simple_split_text(lines[i].strip()))
                    .replace("$ t $", "$T$")
                    .replace("$ T $", "$T$")
                )
        else:
            lines[i] = lines[i].strip()


def convert_atepc(fname, use_tokenizer):
    """
    Converts the input file to the Aspect Term Extraction and Polarity Classification (ATEPC) format.
    :param fname: filename
    :param use_tokenizer: whether to use a tokenizer
    """

    fprint("coverting {} to {}.atepc".format(fname, fname))
    dist_fname = fname.replace("apc_datasets", "atepc_datasets")

    if not os.path.exists(os.path.dirname(dist_fname)) and not os.path.isfile(
        dist_fname
    ):
        os.makedirs(os.path.dirname(dist_fname))
    dist_fname += ".atepc"
    lines = []
    samples = assemble_aspects(fname, use_tokenizer)

    for sample in samples:
        for token_index in range(len(sample[1])):
            token, label, polarity = (
                sample[0].split()[token_index],
                sample[1][token_index],
                sample[2][token_index],
            )
            lines.append(token + " " + label + " " + str(polarity))
        lines.append("\n")

    fout = open(dist_fname, "w", encoding="utf8")
    for line in lines:
        fout.writelines((line + "\n").replace("\n\n", "\n"))
    fout.close()