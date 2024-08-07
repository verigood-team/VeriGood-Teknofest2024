import copy

from mini_pyabsa.framework.configuration_class.configuration_template import ConfigManager
from mini_pyabsa.AspectSentimentTripletExtraction.models.model import EMCGCN

_aste_config_template = {
    "model": EMCGCN,
    "task": "triplet",
    "optimizer": "",
    "learning_rate": 1e-3,
    "cache_dataset": True,
    "warmup_step": -1,
    "deep_ensemble": False,
    "use_bert_spc": True,
    "max_seq_len": 120,
    "patience": 99999,
    "sigma": 0.3,
    "dropout": 0,
    "l2reg": 0.000001,
    "num_epoch": 10,
    "batch_size": 16,
    "seed": 52,
    "output_dim": 3,
    "log_step": 10,
    "dynamic_truncate": True,
    "srd_alignment": True,  # for srd_alignment
    "evaluate_begin": 0,
    "similarity_threshold": 1,  # disable same text check for different examples
    "cross_validate_fold": -1,
    "use_amp": False,
    "overwrite_cache": False,
    "epochs": 100,
    "adam_epsilon": 1e-8,
    "weight_decay": 0.0,
    "emb_dropout": 0.5,
    "num_layers": 1,
    "pooling": "avg",
    "gcn_dim": 300,
    "relation_constraint": True,
    "symmetry_decoding": False,
    "load_cache_path": ""
}

_aste_config_multilingual = {
    "model": EMCGCN,
    "optimizer": "adamw",
    "learning_rate": 0.00002,
    "pretrained_bert": "microsoft/mdeberta-v3-base",
    "use_bert_spc": True,
    "cache_dataset": True,
    "warmup_step": -1,
    "deep_ensemble": False,
    "patience": 99999,
    "max_seq_len": 80,
    "SRD": 3,
    "dlcf_a": 2,  # the a in dlcf_dca_bert
    "dca_p": 1,  # the p in dlcf_dca_bert
    "dca_layer": 3,  # the layer in dlcf_dca_bert
    "use_syntax_based_SRD": False,
    "sigma": 0.3,
    "lcf": "cdw",
    "lsa": False,
    "window": "lr",
    "eta": 1,
    "eta_lr": 0.1,
    "dropout": 0.5,
    "l2reg": 0.000001,
    "num_epoch": 10,
    "batch_size": 16,
    "initializer": "xavier_uniform_",
    "seed": 52,
    "output_dim": 3,
    "log_step": 5,
    "dynamic_truncate": True,
    "srd_alignment": True,  # for srd_alignment
    "evaluate_begin": 0,
    "similarity_threshold": 1,  # disable same text check for different examples
    "cross_validate_fold": -1,
    "load_cache_path": ""
    # split train and test datasets into 5 folds and repeat 3 trainer
}

class ASTEConfigManager(ConfigManager):
    def __init__(self, args, **kwargs):
        """
        Available Params:   {'model': None,
                            'optimizer': "",
                            'learning_rate': 0.00002,
                            'pretrained_bert': "yangheng/deberta-v3-base-absa-v1.1",
                            'cache_dataset': True,
                            'warmup_step': -1,
                            'deep_ensemble': False,
                            'patience': 99999,
                            'use_bert_spc': True,
                            'max_seq_len': 80,
                            'SRD': 3,
                            'lsa': False,
                            'dlcf_a': 2,  # the a in dlcf_dca_bert
                            'dca_p': 1,  # the p in dlcf_dca_bert
                            'dca_layer': 3,  # the layer in dlcf_dca_bert
                            'use_syntax_based_SRD': False,
                            'sigma': 0.3,
                            'lcf': "cdw",
                            'window': "lr",
                            'eta': 1,
                            'eta_lr': 0.1,
                            'dropout': 0,
                            'l2reg': 0.000001,
                            'num_epoch': 10,
                            'batch_size': 16,
                            'initializer': 'xavier_uniform_',
                            'seed': {52, 214}
                            'output_dim': 3,
                            'log_step': 10,
                            'dynamic_truncate': True,
                            'srd_alignment': True,  # for srd_alignment
                            'evaluate_begin': 0,
                            'similarity_threshold': 1,  # disable same text check for different examples
                            'cross_validate_fold': -1   # split train and test datasets into 5 folds and repeat 3 trainer
                            }
        :param args:
        :param kwargs:
        """
        super().__init__(args, **kwargs)

    @staticmethod
    def set_aste_config(configType: str, newitem: dict):
        if isinstance(newitem, dict):
            if configType == "template":
                _aste_config_template.update(newitem)
            elif configType == "multilingual":
                _aste_config_multilingual.update(newitem)

            else:
                raise ValueError(
                    "Wrong value of configuration_class type supplied, please use one from following type: template, base, english, chinese, multilingual, glove, bert_baseline"
                )
        else:
            raise TypeError(
                "Wrong type of new configuration_class item supplied, please use dict e.g.{'NewConfig': NewValue}"
            )

    @staticmethod
    def get_aste_config_multilingual():
        _aste_config_template.update(_aste_config_multilingual)
        return ASTEConfigManager(copy.deepcopy(_aste_config_template))
