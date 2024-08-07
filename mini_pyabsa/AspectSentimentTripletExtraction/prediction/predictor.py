import os
import pickle
import string
from typing import Union

import torch
from findfile import find_file

from mini_pyabsa.utils.data_utils.dataset_manager import detect_infer_dataset
from torch import nn
from tqdm import tqdm

from mini_pyabsa.framework.flag_class.flag_template import DeviceTypeOption

from mini_pyabsa.utils.pyabsa_utils import fprint, set_device, print_args

from mini_pyabsa.framework.flag_class.flag_template import TaskCodeOption

from mini_pyabsa.framework.prediction_class.predictor_template import InferenceModel
from mini_pyabsa.AspectSentimentTripletExtraction.dataset_utils.data_utils_for_inference import (
    ASTEInferenceDataset,
)
from mini_pyabsa.AspectSentimentTripletExtraction.dataset_utils.aste_utils import (
    DataIterator,
    Metric,
)


class AspectSentimentTripletExtractor(InferenceModel):
    task_code = TaskCodeOption.Aspect_Sentiment_Triplet_Extraction

    def __init__(self, checkpoint=None, **kwargs):
        super().__init__(checkpoint, task_code=self.task_code, **kwargs)

        # load from a trainer
        if self.checkpoint and not isinstance(self.checkpoint, str):
            fprint("Load sentiment classifier from trainer")
            self.model = self.checkpoint[0]
            self.config = self.checkpoint[1]
            self.tokenizer = self.checkpoint[2]
        else:
            # load from a model path
            try:
                if "fine-tuned" in self.checkpoint:
                    raise ValueError(
                        "Do not support to directly load a fine-tuned model, please load a .state_dict or .model instead!"
                    )
                fprint("Load sentiment classifier from", self.checkpoint)

                state_dict_path = find_file(
                    self.checkpoint, ".state_dict", exclude_key=["__MACOSX"]
                )
                model_path = find_file(
                    self.checkpoint, ".model", exclude_key=["__MACOSX"]
                )
                tokenizer_path = find_file(
                    self.checkpoint, ".tokenizer", exclude_key=["__MACOSX"]
                )
                config_path = find_file(
                    self.checkpoint, ".config", exclude_key=["__MACOSX"]
                )

                fprint("config: {}".format(config_path))
                fprint("state_dict: {}".format(state_dict_path))
                fprint("model: {}".format(model_path))
                fprint("tokenizer: {}".format(tokenizer_path))

                with open(config_path, mode="rb") as f:
                    self.config = pickle.load(f)
                    self.config.from_checkpoint = checkpoint
                    self.config.auto_device = kwargs.get("auto_device", True)
                    set_device(self.config, self.config.auto_device)

                if state_dict_path or model_path:
                    if state_dict_path:
                        self.model = self.config.model(config=self.config).to(
                            self.config.device
                        )
                        self.model.load_state_dict(
                            torch.load(
                                state_dict_path,
                                map_location=torch.device("cpu"),
                            ),
                            strict=False,
                        )
                    elif model_path:
                        self.model = torch.load(
                            model_path, map_location=DeviceTypeOption.CPU
                        )

                self.tokenizer = self.config.tokenizer

                if kwargs.get("verbose", False):
                    fprint("Config used in Training:")
                    print_args(self.config)

            except Exception as e:
                raise RuntimeError(
                    "Fail to load the model from {}! "
                    "Please make sure the version of checkpoint and PyABSA are compatible."
                    " Try to remove he checkpoint and download again"
                    " \nException: {} ".format(checkpoint, e)
                )
            
        self.dataset = ASTEInferenceDataset(self.config, self.tokenizer)
        self.__post_init__(**kwargs)

    def batch_infer(
        self,
        target_file=None,
        print_result=True,
        save_result=False,
        ignore_error=True,
        **kwargs
    ):
        """
        A deprecated version of batch_predict method.

        Args:
            target_file (str): the path to the target file for inference
            print_result (bool): whether to print the result
            save_result (bool): whether to save the result
            ignore_error (bool): whether to ignore the error

        Returns:
            result (dict): a dictionary of the results
        """
        return self.batch_predict(
            target_file=target_file,
            print_result=print_result,
            save_result=save_result,
            ignore_error=ignore_error,
            **kwargs
        )

    def infer(self, text: str = None, print_result=True, ignore_error=True, **kwargs):
        """
        A deprecated version of the predict method.

        Args:
            text (str): the text to predict
            print_result (bool): whether to print the result
            ignore_error (bool): whether to ignore the error

        Returns:
            result (dict): a dictionary of the results
        """
        return self.predict(
            text=text, print_result=print_result, ignore_error=ignore_error, **kwargs
        )

    def batch_predict(
        self,
        target_file=None,
        print_result=True,
        save_result=False,
        ignore_error=True,
        **kwargs
    ):
        """
        Predict the sentiment from a file of sentences.
        param: target_file: the file path of the sentences to be predicted.
        param: print_result: whether to print the result.
        param: save_result: whether to save the result.
        param: ignore_error: whether to ignore the error when predicting.
        param: kwargs: other parameters.
        """
        self.config.eval_batch_size = kwargs.get("eval_batch_size", 32)

        save_path = os.path.join(
            os.getcwd(),
            "{}.{}.result.json".format(
                self.config.task_name, self.config.model.__name__
            ),
        )

        target_file = detect_infer_dataset(
            target_file, task_code=TaskCodeOption.Aspect_Sentiment_Triplet_Extraction
        )
        if not target_file:
            raise FileNotFoundError("Can not find inference datasets!")

        self.dataset.prepare_infer_dataset(target_file, ignore_error=ignore_error)

        return self._run_prediction(
            save_path=save_path if save_result else None, print_result=print_result
        )

    def predict(
        self,
        text: Union[str, list] = None,
        print_result=True,
        ignore_error=True,
        **kwargs
    ):
        """
        Predict the sentiment from a sentence or a list of sentences.
        param: text: the sentence to be predicted.
        param: print_result: whether to print the result.
        param: ignore_error: whether to ignore the error when predicting.
        param: kwargs: other parameters.
        """
        self.config.eval_batch_size = kwargs.get("eval_batch_size", 32)
        if text:
            self.dataset.prepare_infer_sample(text, ignore_error=ignore_error)
        else:
            self.dataset.prepare_infer_sample("", ignore_error=ignore_error)
        if isinstance(text, str):
            try:
                entity_list, results = self._run_prediction(print_result=print_result, **kwargs)
                return entity_list, results
            except Exception as e:
                return {
                    "text": text,
                    "Triplets": "[]",
                    "error": str(e),
                    "error_type": "RuntimeError",
                }
        else:
            entity_list, results = self._run_prediction(print_result=print_result, **kwargs)
            if print_result:
                print("Entity List:", entity_list)
                print("Results:", results)
            return entity_list, results

    def _run_prediction(self, save_path=None, print_result=True, **kwargs):
        self.model.eval()
        all_results = []
        with torch.no_grad():
            data_loader = DataIterator(
                self.dataset.convert_examples_to_features(), self.config
            )
            if len(self.dataset) > 1:
                it = tqdm(data_loader, desc="Predicting")
            else:
                it = data_loader
            for i, batch in enumerate(it):
                (
                    sentence_ids,
                    sentences,
                    token_ids,
                    lengths,
                    masks,
                    sens_lens,
                    token_ranges,
                    aspect_tags,
                    tags,
                    word_pair_position,
                    word_pair_deprel,
                    word_pair_pos,
                    word_pair_synpost,
                    tags_symmetry,
                ) = batch

                inputs = {
                    "token_ids": token_ids,
                    "masks": masks,
                    "word_pair_position": word_pair_position,
                    "word_pair_deprel": word_pair_deprel,
                    "word_pair_pos": word_pair_pos,
                    "word_pair_synpost": word_pair_synpost,
                }

                preds = self.model(inputs)[-1]
                preds = nn.functional.softmax(preds, dim=-1)
                preds = torch.argmax(preds, dim=3)

                metric = Metric(
                    self.config,
                    preds,
                    tags,
                    lengths,
                    sens_lens,
                    token_ranges,
                )

                new_result = {
                    "sentence_id": "",
                    "sentence": "",
                    "Triplets": [],
                    "True Triplets": [],
                }

                try:
                    results = metric.parse_triplet(golden=True)
                except Exception as e:
                    results = metric.parse_triplet(golden=False)

                for j, triplets in enumerate(results[1]):
                    new_result["sentence_id"] = sentence_ids[j]
                    new_result["sentence"] = sentences[j]

                    for k, triplet in enumerate(triplets):
                        asp_head, asp_tail, opn_head, opn_tail, polarity = triplet
                        triplet = {
                            "Aspect": " ".join(
                                sentences[j].split()[asp_head : asp_tail + 1]
                            ),
                            "Opinion": " ".join(
                                sentences[j].split()[opn_head : opn_tail + 1]
                            ),
                            "Polarity": self.config.index_to_label[polarity],
                        }

                        if triplet["Aspect"] in string.punctuation:
                            triplet["Aspect"] = " "

                        new_result["Triplets"].append(triplet)

                    all_results.append(new_result)


            for result in all_results:
                fprint("Batch: {}".format(i), result)

            entity_list, results = find_myresult(self, all_results)
            return entity_list, results

    def clear_input_samples(self):
        self.dataset.all_data = []


class Predictor(AspectSentimentTripletExtractor):
    pass


def preprocessing_for_results(self, all_results):

    suffixes = ["gen","sal","sel","gıl","gil", "mız", "miz", "muz", "müz", "nuz", "nüz", "nız", "niz", "ca", "ce", "ça","çe","da","de", "dı", "di", "du", "dü", "ta","te", "tı", "ti", "tu", "tü", "lu","lü", "li","lı","la","le","ki", "n", "m","ş","s", "y", "k", "a","e","r","ı","i","u","ü", "z"]
    control = ["@",",",".","!","?","#","*"] 

    for _item in all_results:
        for triplet in _item["Triplets"]:
            
            processing_word = triplet["Aspect"]
            while len(processing_word) > 0 and processing_word[0] in control:
                processing_word = processing_word[1:].strip()
            while len(processing_word) > 0 and processing_word[-1] in control:
                processing_word = processing_word[:-1].strip()
            
            triplet["Aspect"] = processing_word

            len_processing_word = len(processing_word.split())
            if len_processing_word > 1:
                splited_words = processing_word.split()
                processing_word = splited_words[-1]
            
            if "'" in processing_word or "’" in processing_word:
                cut_word = processing_word
                i=0
                while i < len(suffixes):
                    if processing_word.endswith(suffixes[i]):
                        cut_word = processing_word[:-len(suffixes[i])]
                        processing_word = cut_word
                        i=0
                    else:
                        i+=1
                
                cut_word = cut_word[:-1]
                if len_processing_word == 1:
                    triplet["Aspect"] = cut_word
                else:
                    temp_word = triplet["Aspect"]
                    splited_temp = temp_word.split()
                    splited_temp[len_processing_word-1] = cut_word
                    splited_temp = splited_temp[:len_processing_word]
                    triplet["Aspect"] = ' '.join(splited_temp)


def find_myresult(self, all_results):

    preprocessing_for_results(self, all_results)
    
    entity_list = []
    results = []

    for item in all_results:
        for triplet in item["Triplets"]:

            new_dict = {}
            new_dict[triplet["Aspect"]] = triplet["Polarity"]
            
            new_key = triplet["Aspect"].lower()
            new_value = triplet["Polarity"].lower()
            
            exists = False
            for res in results:
                res_key = list(res.keys())[0].lower()
                res_value = list(res.values())[0].lower()
                if res_key == new_key and res_value == new_value:
                    exists = True
                    break
            
            if not exists:
                results.append(new_dict)

    for res in results:
        key = list(res.keys())[0]
        res_value = list(res.values())[0]

        if res_value == "Positive":
            res_value = "olumlu"
        elif res_value == "Negative":
            res_value = "olumsuz"
        elif res_value == "Neutral":
            res_value = "nötr"
        
        res[key] = res_value
    
    key_polarities = {}
    original_keys = {}

    for d in results:
        original_key = list(d.keys())[0]
        key_lower = original_key.lower()
        value = d[original_key]
        
        if key_lower not in original_keys:
            original_keys[key_lower] = original_key
        
        if key_lower in key_polarities:
            key_polarities[key_lower].append(value)
        else:
            key_polarities[key_lower] = [value]
    
    filtered_results = []
    for key, values in key_polarities.items():
        if len(values) > 1 and "nötr" in values:
            values.remove("nötr")
        for value in values:
            filtered_results.append({original_keys[key]: value})

    results = filtered_results


    for res in results:
        res_key = list(res.keys())[0]
        if res_key not in entity_list:
            entity_list.append(res_key)

    return entity_list, results