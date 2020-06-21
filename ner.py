import logging

from typing import List, Optional
from transformers import PreTrainedTokenizer, AutoTokenizer, AutoModelForTokenClassification, DefaultDataCollator
from dataclasses import dataclass
from transformers.optimization import AdamW

import torch
from os import path
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torch import nn
import sys

from tqdm import tqdm


@dataclass
class InputExample:
    guid: str
    words: List[str]
    labels: Optional[List[str]]


@dataclass
class InputFeature:
    input_ids: List[int]
    attention_mask: List[int]
    token_type_ids: List[int]
    label_ids: List[int]

class NERDataset(Dataset):

    def __init__(self, filename):
        self.features = torch.load(filename)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, item) -> InputFeature:
        return self.features[item]
#        return self.features[item].__dict__


logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.addHandler(handler)

PAD_TOKEN = 0
PAD_MASK_TOKEN = 0
PAD_TOKEN_LABEL_ID = nn.CrossEntropyLoss().ignore_index
PAD_TOKEN_SEGMENT_ID = 0
ITERATION = 1000

MODEL_NAME = "xlm-roberta-base"


def read_examples_from_file(filename) -> List[InputExample]:
    examples = []
    words = []
    labels = []
    guid_index = 0
    with open(filename) as f:
        logger.info("Create dataset")
        for line in f:
            line = line.rstrip()
            if line:
                splitted_line = line.split()
                assert len(splitted_line) == 2
                words.append(splitted_line[0])
                labels.append(splitted_line[1])
            else:
                assert len(words) == len(labels)
                examples.append(InputExample(guid=f"{guid_index}", words=words, labels=labels))
                words = []
                labels = []
                guid_index += 1
        if words:
            assert len(words) == len(labels)
            examples.append(InputExample(guid=f"{guid_index}", words=words, labels=labels))
    return examples


def convert_examples_to_features(examples: List[InputExample], label_list: List[str], tokenizer: PreTrainedTokenizer,
                                 max_seq_length: int, sep_token="[SEP]",
                                 sequence_a_segment_id=0):
    label_map = {label: i for i, label in enumerate(label_list)}
    features = []
    for example in examples:
        tokens = []
        label_ids = []
        for word, label in zip(example.words, example.labels):
            word_tokens = tokenizer.tokenize(word)
            assert len(word_tokens) > 0
            label_ids.extend([label_map[label]] + [PAD_TOKEN_LABEL_ID] * (len(word_tokens) - 1))
            tokens.extend(word_tokens)
        special_tokens_count = tokenizer.num_special_tokens_to_add()
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[:(max_seq_length - special_tokens_count)]
            label_ids = label_ids[:(max_seq_length - special_tokens_count)]
        # Have to pad the ending token
        tokens += [sep_token]
        label_ids += [PAD_TOKEN_LABEL_ID]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)

        input_ids += [PAD_TOKEN] * padding_length
        input_mask += [PAD_MASK_TOKEN] * padding_length
        segment_ids += [PAD_TOKEN_SEGMENT_ID] * padding_length
        label_ids += [PAD_TOKEN_LABEL_ID] * padding_length
        # WTF {
        # if "token_type_ids" not in tokenizer.model_input_names:
        #     segment_ids = None
        # } WTF
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length
        features.append(
            InputFeature(input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids,
                         label_ids=label_ids))
    return features


for filename in ['train.txt', 'dev.txt', 'test.txt']:
    result_filename, _ = path.splitext(filename)
    if not path.exists(f"{result_filename}.ft"):
        examples = read_examples_from_file(filename)
        with open("labels.txt") as f:
            labels_list = f.read().splitlines()

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        features = convert_examples_to_features(examples, tokenizer=tokenizer, label_list=labels_list,
                                                max_seq_length=32)
        torch.save(features, f"{result_filename}.ft")

# train
train_dataset = NERDataset("dev.ft")
logger.info("Done create dataset")
train_sampler = RandomSampler(train_dataset)
default_collator = DefaultDataCollator()
train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=32, collate_fn=default_collator.collate_batch)
model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)

# optimizer
optimizer = AdamW(model.parameters())

# load from_pretrained, finetune
model.train()
for _ in range(ITERATION):
    for step, data in enumerate(tqdm(train_loader)):
        model.zero_grad()
        #loss, prediction_scores, seq_relationship_score, hidden_states, attentions = model(**data)
        loss, prediction_scores  = model(**data)
        loss.backward()
        optimizer.step()
