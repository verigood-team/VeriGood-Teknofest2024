import re
from collections import Counter, OrderedDict

import numpy as np
import tqdm

import os
import spacy
import termcolor

from ..dataset_utils.aste_utils import (
    VocabHelp,
    Instance,
)

from mini_pyabsa.utils.file_utils.file_utils import load_dataset_from_file
from mini_pyabsa.utils.pyabsa_utils import fprint

def configure_spacy_model(config):
    if not hasattr(config, "spacy_model"):
        config.spacy_model = "en_core_web_sm"
    global nlp
    try:
        nlp = spacy.load(config.spacy_model)
    except:
        fprint(
            "Can not load {} from spacy, try to download it in order to parse syntax tree:".format(
                config.spacy_model
            ),
            termcolor.colored(
                "\npython -m spacy download {}".format(config.spacy_model), "green"
            ),
        )
        try:
            os.system("python -m spacy download {}".format(config.spacy_model))
            nlp = spacy.load(config.spacy_model)
        except:
            raise RuntimeError(
                "Download failed, you can download {} manually.".format(
                    config.spacy_model
                )
            )
    return nlp


class ASTEInferenceDataset:
    syn_post_vocab = None
    postag_vocab = None
    deprel_vocab = None
    post_vocab = None
    syn_post_vocab = None
    token_vocab = None

    all_tokens = []
    all_deprel = []
    all_postag = []
    all_postag_ca = []
    all_max_len = []

    labels = [
        "N",
        "B-A",
        "I-A",
        "A",
        "B-O",
        "I-O",
        "O",
        "Negative",
        "Neutral",
        "Positive",
    ]
    label_to_index, index_to_label = OrderedDict(), OrderedDict()
    for i, v in enumerate(labels):
        label_to_index[v] = i
        index_to_label[i] = v

    def prepare_infer_sample(self, text, ignore_error=True):
        if isinstance(text, str):
            text = [text]
        _data = []
        for i in range(len(text)):
            if not text[i].count("####"):
                text[i] = text[i].strip() + " [none] " + "####[]"
        self.process_data(text, ignore_error)


    def process_data(self, samples, ignore_error=True):
        
        sentence = ""
        self.data = []
        if len(samples) > 1:
            it = tqdm.tqdm(samples, desc="preparing dataloader")
        else:
            it = samples
        # record polarities type to update output_dim
        label_set = set()
        for ex_id, sample in enumerate(it):
            try:
                if samples[ex_id].count("####"):
                    sentence, annotations = samples[ex_id].split("####")
                elif samples[ex_id].count("$LABEL$"):
                    sentence, annotations = samples[ex_id].split("$LABEL$")
                else:
                    raise ValueError(
                        "Invalid annotations format, please check your dataset file."
                    )

                sentence, annotations = self.update_for_spacy(sentence, annotations)

                sentence = self.truncate_sentence(sentence).strip()
                
                annotations = eval(annotations.strip())
                
                annotations = self.truncate_annotations(annotations, len(sentence.split()))

                sentence = sentence.replace(" - ", " placeholder ").replace("-", " ")
                prepared_data = self.get_syntax_annotation(sentence, annotations)
                tokens, deprel, postag, postag_ca, max_len = load_tokens(prepared_data)
                self.all_tokens.extend(tokens)
                self.all_deprel.extend(deprel)
                self.all_postag.extend(postag)
                self.all_postag_ca.extend(postag_ca)
                self.all_max_len.append(max_len)
                prepared_data["id"] = ex_id
                prepared_data["sentence"] = sentence.replace("placeholder", "-")

                for annotation in annotations:
                    label_set.add(annotation[-1])

                self.data.append(prepared_data)

            except Exception as e:
                if ignore_error:
                    fprint(
                        "Ignore error while processing: {} Error info:{}".format(
                            sentence, e
                        )
                    )
                else:
                    raise RuntimeError(
                        "Ignore error while processing: {} Catch Exception: {}, use ignore_error=True to remove error samples.".format(
                            sentence, e
                        )
                    )

    def __init__(self, config, tokenizer, dataset_type="train"):
        self.data = None
        self.nlp = configure_spacy_model(config)
        self.config = config
        self.tokenizer = tokenizer
        self.dataset_type = dataset_type
        self.config.label_to_index = self.label_to_index
        self.config.index_to_label = self.index_to_label
        self.config.output_dim = len(self.label_to_index)


    def __len__(self):
        return len(self.data)

    def convert_examples_to_features(self, **kwargs):
        _data = []
        if len(self.data) > 1:
            it = tqdm.tqdm(self.data, desc="converting data to features")
        else:
            it = self.data
        for data in it:
            try:
                feat = Instance(
                    self.tokenizer,
                    data,
                    self.config.post_vocab,
                    self.config.deprel_vocab,
                    self.config.postag_vocab,
                    self.config.syn_post_vocab,
                    self.config,
                )
                _data.append(feat)
            except IndexError as e:
                if kwargs.get("ignore_error", True):
                    fprint(
                        "Ignore error while processing: {} Error info:{}".format(
                            data["sentence"], e
                        )
                    )
                else:
                    raise RuntimeError(
                        "Ignore error while processing: {} Catch Exception: {}, use ignore_error=True to remove error samples.".format(
                            data["sentence"], e
                        )
                    )
        self.data = _data
        return self.data

    def get_syntax_annotation(self, sentence, annotation):

        # Extract aspect and opinion terms from annotation
        aspect_spans = [
            (aspect_span[0], aspect_span[-1]) for (aspect_span, _, _) in annotation
        ]
        opinion_spans = [
            (opinion_span[0], opinion_span[-1]) for (_, opinion_span, _) in annotation
        ]
        sentiments = [sentiment_label for (_, _, sentiment_label) in annotation]

        # Tokenize sentence
        # tokens = re.findall(r'\w+|[^\w\s]', sentence)
        # tokens = sentence.split()
        tokens = [token.text for token in self.nlp(sentence)]
        postags, heads, deprels = self.get_dependencies(tokens)

        # Generate triples
        triples = []

        for i, aspect_span in enumerate(aspect_spans):
            for j, opinion_span in enumerate(opinion_spans):
                if aspect_span == opinion_span:
                    continue
                aspect_start, aspect_end = aspect_span
                opinion_start, opinion_end = opinion_span
                # if aspect_start > opinion_start:
                #     aspect_start, opinion_start = opinion_start, aspect_start
                #     aspect_end, opinion_end = opinion_end, aspect_end
                # if aspect_end >= opinion_start:
                #     continue
                uid = f"{i}-{j}"
                target_tags = generate_tags(tokens, aspect_start, aspect_end, "BIO")
                opinion_tags = generate_tags(tokens, opinion_start, opinion_end, "BIO")
                triples.append(
                    {
                        "uid": uid,
                        "target_tags": target_tags,
                        "opinion_tags": opinion_tags,
                        "sentiment": sentiments[j]
                        .replace("POS", "Positive")
                        .replace("NEG", "Negative")
                        .replace("NEU", "Neutral"),
                    }
                )

        # Generate output dictionary
        output = {
            "id": "",
            "sentence": sentence,
            "postag": postags,
            "head": heads,
            "deprel": deprels,
            "triples": triples,
        }

        return output


    def get_dependencies(self, tokens):
        
        # Replace special characters in tokens with placeholders
        placeholder_tokens = []
        for token in tokens:
            if re.search(r"[^\w\s]", token):
                placeholder = f"__{token}__"
                placeholder_tokens.append(placeholder)
            else:
                placeholder_tokens.append(token)

        # Get part-of-speech tags and dependencies using spaCy
        doc = self.nlp(" ".join(tokens))
        postags = [token.pos_ for token in doc]
        heads = [token.head.i for token in doc]
        deprels = [token.dep_ for token in doc]

        return postags, heads, deprels
    
    def truncate_sentence(self, sentence):

        encoded = self.tokenizer.encode(
                    sentence,
                    padding="do_not_pad",
                    max_length=self.config.max_seq_len,
                    truncation=True,
                )
        
        if len(encoded) == 256:
            none_label = self.tokenizer.encode(" [ none ]", padding="do_not_pad")[1:-1]
            encoded = encoded[:-6] + none_label + encoded[-1:]
            
            
        decoded = self.tokenizer.decode(encoded)
        decoded = decoded[5: -5].strip()

        spacy_doc = self.nlp(decoded)
        spacy_tokens = [token.text for token in spacy_doc]

        sentence = " ".join(spacy_tokens)

        return sentence
    
    def truncate_annotations(self, annotations, senLength):
        
        for i in range(len(annotations) - 1, -1, -1):
            
            annotation = annotations[i]
            remove = False

            for idx in annotation[0]:
                if idx >= senLength:
                    remove = True
                    break

            if remove:
                del annotations[i]

            else:
                remove = False
                for idx in annotation[1]:
                    if idx >= senLength:
                        remove = True
                        break

                if remove:
                    del annotations[i]
        
        return annotations

    def update_for_spacy(self, sentence, annotations):
        
        sentence = sentence.strip()
        base_tokens = sentence.split()
        
        spacy_doc = self.nlp(sentence)
        spacy_tokens = [token.text for token in spacy_doc]

        annotations = eval(annotations)
            
        i=0
        pair_list = []
        for token in base_tokens:
            
            temp_list = []
            while spacy_tokens[i] in token:
                
                temp_list.append(i)
                token = token[len(spacy_tokens[i]):].lstrip()
                i+=1

                if i == len(spacy_tokens):
                    break

            pair_list.append(temp_list)

        updated_annotations = []
        for i, annotation in enumerate(annotations):

            for j in range(len(annotation[0])):
            
                idx = annotation[0][j]
                annotation[0][j] = pair_list[idx]

            entity_idx = [item for sublist in annotation[0] for item in sublist]

            for j in range(len(annotation[1])):
                
                idx = annotation[1][j]
                annotation[1][j] = pair_list[idx]

            opinion_idx = [item for sublist in annotation[1] for item in sublist]
            
            updated_annotations.append((entity_idx, opinion_idx, annotation[2]))

        sentence = " ".join(spacy_tokens)
        annotations = updated_annotations

        return sentence, str(annotations)

def generate_tags(tokens, start, end, scheme):

    if scheme == "BIO":
        tags = ["O"] * len(tokens)
        tags[start] = "B"
        for i in range(start + 1, end + 1):
            tags[i] = "I"
        return " ".join([f"{token}\\{tag}" for token, tag in zip(tokens, tags)])
    elif scheme == "IOB2":
        tags = ["O"] * len(tokens)
        tags[start] = "B"
        for i in range(start + 1, end + 1):
            tags[i] = "I"
        if end < len(tokens) - 1 and tags[end + 1] == "I":
            tags[end] = "B"
        return " ".join([f"{token}\\{tag}" for token, tag in zip(tokens, tags)])
    else:
        raise ValueError(f"Invalid tagging scheme '{scheme}'.")


def load_tokens(data):

    tokens = []
    deprel = []
    postag = []
    postag_ca = []

    max_len = 0
    sentence = data["sentence"].split()
    tokens.extend(sentence)
    deprel.extend(data["deprel"])
    postag_ca.extend(data["postag"])
    # postag.extend(d['postag'])
    n = len(data["postag"])
    tmp_pos = []
    for i in range(n):
        for j in range(n):
            tup = tuple(sorted([data["postag"][i], data["postag"][j]]))
            tmp_pos.append(tup)
    postag.extend(tmp_pos)

    max_len = max(len(sentence), max_len)
    return tokens, deprel, postag, postag_ca, max_len