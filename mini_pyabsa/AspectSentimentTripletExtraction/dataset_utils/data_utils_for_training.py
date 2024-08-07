import os
from collections import Counter, OrderedDict
from multiprocessing import Pool

import tqdm

import os
import spacy
import termcolor

from mini_pyabsa.framework.dataset_class.dataset_template import PyABSADataset
from mini_pyabsa.AspectSentimentTripletExtraction.dataset_utils.aste_utils import (
    VocabHelp,
    Instance,
    load_tokens,
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

def generate_tags(tokens, start, end, scheme):
    # print('Generating tags for tokens: ', tokens)
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


class ASTEDataset(PyABSADataset):
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


    def load_data_from_file(self, file_path, **kwargs):

        lines = load_dataset_from_file(
            self.config.dataset_file[self.dataset_type], config=self.config
        )

        all_data = []
        # record polarities type to update output_dim
        label_set = set()

        for ex_id in tqdm.tqdm(range(0, len(lines)), desc="preparing dataloader"):
            try:
                if lines[ex_id].count("####"):
                    sentence, annotations = lines[ex_id].split("####")
                elif lines[ex_id].count("$LABEL$"):
                    sentence, annotations = lines[ex_id].split("$LABEL$")
                else:
                    raise ValueError(
                        "Invalid annotations format, please check your dataset file."
                    )

                sentence, annotations = self.update_for_spacy(sentence, annotations) 
                sentence = self.truncate_sentence(sentence).strip()
                annotations = eval(annotations.strip())
                annotations = self.truncate_annotations(annotations, len(sentence.split()))
                sentence = sentence.replace(" - ", " placeholder ").replace("-", " ")

                try:
                    prepared_data = self.get_syntax_annotation(sentence, annotations)
                except Exception as e:
                    print(sentence)
                    print(lines[ex_id])
                    print(annotations)
                    continue
                prepared_data["id"] = ex_id
                tokens, deprel, postag, postag_ca, max_len = load_tokens(prepared_data)
                self.all_tokens.extend(tokens)
                self.all_deprel.extend(deprel)
                self.all_postag.extend(postag)
                self.all_postag_ca.extend(postag_ca)
                self.all_max_len.append(max_len)
                label_set.add(annotation[-1] for annotation in annotations)
                prepared_data["sentence"] = sentence.replace("placeholder", "-")
                all_data.append(prepared_data)
            except Exception as e:
                print(e)
                continue
        self.data = all_data

    def __init__(self, config, tokenizer, dataset_type="train"):
        self.nlp = configure_spacy_model(config)
        super().__init__(config=config, tokenizer=tokenizer, dataset_type=dataset_type)
        self.config.label_to_index = self.label_to_index
        self.config.index_to_label = self.index_to_label
        self.config.output_dim = len(self.label_to_index)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def convert_examples_to_features(self):

        self.get_vocabs()
        _data = []
        for data in tqdm.tqdm(self.data, desc="converting data to features"):
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
            except Exception as e:
                fprint(
                    "Processing error for: {}. Exception: {}".format(
                        data["sentence"], e
                    )
                )
        self.data = _data

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
        tokens = sentence.split()

        postags, heads, deprels = self.get_dependencies(tokens)

        # Generate triples
        triples = []
        for i, aspect_span in enumerate(aspect_spans):
        
            if aspect_span == opinion_spans[i]:
                continue
            aspect_start, aspect_end = aspect_span
            opinion_start, opinion_end = opinion_spans[i]
            # if aspect_start > opinion_start:
            #     aspect_start, opinion_start = opinion_start, aspect_start
            #     aspect_end, opinion_end = opinion_end, aspect_end
            # if aspect_end >= opinion_start:
            #     continue
            uid = f"{i}"
            target_tags = generate_tags(tokens, aspect_start, aspect_end, "BIO")
            opinion_tags = generate_tags(tokens, opinion_start, opinion_end, "BIO")
            triples.append(
                {
                    "uid": uid,
                    "target_tags": target_tags,
                    "opinion_tags": opinion_tags,
                    "sentiment": sentiments[i]
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
        # for token in tokens:
        #     if re.search(r"[^\w\s]", token):
        #         placeholder = f"__{token}__"
        #         placeholder_tokens.append(placeholder)
        #     else:
        #         placeholder_tokens.append(token)

        # Get part-of-speech tags and dependencies using spaCy
        doc = self.nlp(" ".join(tokens))
        postags = [token.pos_ for token in doc]
        heads = [token.head.i for token in doc]
        deprels = [token.dep_ for token in doc]

        return postags, heads, deprels

    def get_vocabs(self):

        if (
            self.config.get("syn_post_vocab") is None
            and self.config.get("postag_vocab") is None
            and self.config.get("deprel_vocab") is None
            and self.config.get("syn_post_vocab") is None
            and self.config.get("token_vocab") is None
        ):
            token_counter = Counter(self.all_tokens)
            deprel_counter = Counter(self.all_deprel)
            postag_counter = Counter(self.all_postag)
            postag_ca_counter = Counter(self.all_postag_ca)
            # deprel_counter['ROOT'] = 1
            deprel_counter["self"] = 1

            max_len = max(self.all_max_len)
            # post_counter = Counter(list(range(-max_len, max_len)))
            post_counter = Counter(list(range(0, max_len)))
            syn_post_counter = Counter(list(range(0, 5)))

            # build vocab
            print("building vocab...")
            token_vocab = VocabHelp(token_counter, specials=["<pad>", "<unk>"])
            post_vocab = VocabHelp(post_counter, specials=["<pad>", "<unk>"])
            deprel_vocab = VocabHelp(deprel_counter, specials=["<pad>", "<unk>"])
            postag_vocab = VocabHelp(postag_counter, specials=["<pad>", "<unk>"])
            syn_post_vocab = VocabHelp(syn_post_counter, specials=["<pad>", "<unk>"])

            self.config.token_vocab = token_vocab
            self.config.post_vocab = post_vocab
            self.config.deprel_vocab = deprel_vocab
            self.config.postag_vocab = postag_vocab
            self.config.syn_post_vocab = syn_post_vocab
            self.config.post_size = len(post_vocab)
            self.config.deprel_size = len(deprel_vocab)
            self.config.postag_size = len(postag_vocab)
            self.config.synpost_size = len(syn_post_vocab)
    
    def truncate_sentence(self, sentence):
        
        encoded = self.tokenizer.encode(
                    sentence,
                    padding="do_not_pad",
                    max_length=self.config.max_seq_len,
                    truncation=True,
                )
        
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