""" Named entity recognition fine-tuning: utilities to work with CLUENER task. """
import torch
import logging
import os
import copy
import math
import json
from .utils_ner import DataProcessor
from tokenizers import AddedToken
from transformers import AutoTokenizer
from .get_entity_label import print_entity_length, get_input_label, get_label, get_label_nested, get_label_all_type
logger = logging.getLogger(__name__)

def write_stat_in_files(all_entity_frequency, data_output_dir, data_type, tokenizer):
    """
    Args:
        all_entity_frequency: {key = entity token, value = [frequency, the original input tokens] }
    Returns:
        rare_entity_dict = {key = rare entity token, value = [frequency,
        id of the rare entity(will be used to get the prediction and label of this entity) }
    """
    rare_entity_dict = {}
    all_entity_frequency_json = []
    for i in all_entity_frequency.keys():
        json_d = {}
        json_d['entity'] = i
        json_d['frequency'] = str(all_entity_frequency[i][0])
        if all_entity_frequency[i][0] <= 3:
            rare_entity_dict[i] = [all_entity_frequency[i][0], tokenizer.encode(i)]
            json_d['the original input'] = all_entity_frequency[i][1]
        all_entity_frequency_json.append(json_d)
    if not os.path.exists(data_output_dir):
        os.makedirs(data_output_dir)

    data_output_dir = os.path.join(data_output_dir, data_type)
    with open(data_output_dir, "w") as writer:
        for record in all_entity_frequency_json:
            writer.write(json.dumps(record) + '\n')
    return rare_entity_dict


class InputExample(object):
    """A single training/test example for token classification."""
    def __init__(self, guid, text_a, labels, entity_type=''):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: list. The words of the sequence.
            labels: (Optional) list. The labels for each word of the sequence. This should be
            specified for train and dev examples, but not for test examples.
            entity_type: special for nested ner, for flat ner, entity_type is set ''
        """
        self.guid = guid
        self.text_a = text_a
        self.labels = labels
        self.entity_type = entity_type
    def __repr__(self):
        return str(self.to_json_string())
    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output
    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self, input_ids, input_mask, input_len, segment_ids, entity,
                 input_words, input_label, entity_type=[-100], input_words_ids=[0]):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.input_len = input_len
        self.entity = entity
        self.entity_type = entity_type
        self.input_words = input_words
        self.input_label = input_label
        self.input_words_ids = input_words_ids

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

def collate_fn(batch):
    """
    batch should be a list of (sequence, target, length) tuples...
    Returns a padded tensor of sequences sorted from longest to shortest,
    """
    all_input_ids, all_attention_mask, all_token_type_ids, all_lens, entity_labels, entity_types, input_labels, all_match_input_ids = map(torch.stack, zip(*batch))
    max_len = max(all_lens).item()
    all_input_ids = all_input_ids[:, :max_len]
    all_attention_mask = all_attention_mask[:, :max_len]
    all_token_type_ids = all_token_type_ids[:, :max_len]
    input_labels = input_labels[:, :max_len]
    return all_input_ids, all_attention_mask, all_token_type_ids, all_lens, entity_labels, entity_types, input_labels, all_match_input_ids

def convert_examples_to_features(english, task_name, lower_case, dataset,  max_len_for_entity,  tokenizer_name, examples,
                        label_list, max_seq_length, tokenizer,pad_on_left=False, pad_token=0, pad_token_segment_id=0,
                        sequence_a_segment_id=0, mask_padding_with_zero=True, extend_token=None, cur_vocab_id=None):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
        cur_vocab_id: train = 50257, eval and test =the output cur_vocab_id from train
        extend_token = dict: key=token, value=id
    """
    count = 0
    all_entity_frequency = {}
    positive_sample = 0
    negative_sample = 0
    sum_length_of_example = 0

    do_not_separate_type = True

    label_map = {label: i for i, label in enumerate(label_list)}
    entity_map = {i: label for i, label in enumerate(label_list) if 'B' in label or 'I' in label}
    for i in entity_map.keys():
        entity_map[i] = entity_map[i].strip('B-')
        entity_map[i] = entity_map[i].strip('I-')

    entity_type_map = {}
    # dict used to record entities: {key: entity_name, value: id for this entity type, in range(num_entity_types)}
    j = 0
    for i in entity_map.values():
        if j % 2 == 0:
            entity_type_map[i] = j//2
        j = j+1


    features = []
    mismatch = 0# 一个entity在同一句话中出现多次, mismatch+1
    if english:
        if dataset in['conll2003', 'conll2003_mrc']:
            entity_num = 4
            type_to_words = {
                'PER': 'person',
                'LOC': 'location',
                'MISC': 'others',
                'ORG': 'organization'
            }
        elif dataset == 'ontonote':
            entity_num = 18
            type_to_words = {
                'NORP': 'nationality',
                'GPE': 'government',
                'FAC': 'buildings',
                'PERSON': 'person',
                'DATE': 'date',
                'ORG': 'organization',
                'LOC': 'location',
                'WORK_OF_ART': 'work_of_art',
                'EVENT': 'event',
                'CARDINAL': 'cardinal',
                'ORDINAL': 'ordinal',
                "PRODUCT": 'product',
                'QUANTITY': 'quantity',
                'TIME': 'time',
                'PERCENT': 'percent',
                'MONEY': 'money',
                'LAW': 'law',
                'LANGUAGE': 'language'
            }

        elif dataset == 'genia':
            entity_num = 5
        elif dataset == 'ace05':
            entity_num = 7
        else:
            raise(NotImplementedError)
        type_count = [0] * entity_num
        sum_entity_length = [0] * entity_num

        print("**********gpt2_english tokenizer, get all the entities in on sequence"
              "(which means duplicate the input sequence k times according to the entity number in this sequence)"
              "one input sequence with one label, construct negative sample ************")
        feature_count = 0
        if dataset in ['genia', 'ace05']:
            entity_type_map = {}
            j = 0
            for i in entity_map.values():
                if j % 2 == 0:
                    entity_type_map[i] = j//2
                j = j+1

            for (ex_index, example) in enumerate(examples):
                if ex_index % 10000 == 0:
                    logger.info("Writing example %d of %d", ex_index, len(examples))
                new_text = ' '.join(example.text_a)
                if lower_case:
                    new_text = new_text.lower()
                    for i in range(len(example.text_a)):
                        example.text_a[i] = example.text_a[i].lower()

                tokens = tokenizer.tokenize(' ' + new_text)# 在每句话开头加上空格，保证第一个单词可以被tokenized as G开头
                sum_length_of_example += len(tokens)

                true_label_place = [k for k in range(len(example.labels)) if example.labels[k] != 'O']
                true_entity = [example.text_a[k] for k in true_label_place]
                true_entity_set = set(true_entity)
                if len(true_entity_set) != len(true_entity):
                    mismatch += 1
                label_ids = [label_map[x] for x in example.labels]
                if len(label_ids) == 0:
                    continue

                input_ids = tokenizer.convert_tokens_to_ids(tokens)
                input_len = min(len(input_ids), max_seq_length)
                if input_len == 0:
                    continue

                # note: input_labels[i] 长度与example.text_a一致，也就是words的长度
                entity_id = get_label_nested(tokenizer, example.labels, input_words=example.text_a,
                                        max_len_for_entity=max_len_for_entity, sum_entity_length =sum_entity_length,
                                        all_entity_frequency=all_entity_frequency, type_count=type_count,
                                        input_ids=input_ids, entity_type_map=entity_type_map, entity_type=example.entity_type)
                special_tokens_count = 0
                if len(input_ids) > max_seq_length - special_tokens_count:
                    input_ids = input_ids[: (max_seq_length - special_tokens_count)]

                segment_ids = [sequence_a_segment_id] * len(input_ids)
                input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

                # Zero-pad up to the sequence length.
                padding_length = max_seq_length - len(input_ids)
                if pad_on_left:
                    input_ids = ([pad_token] * padding_length) + input_ids
                    input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
                    segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
                else:
                    input_ids += [pad_token] * padding_length
                    input_mask += [0 if mask_padding_with_zero else 1] * padding_length
                    segment_ids += [pad_token_segment_id] * padding_length

                assert len(input_ids) == max_seq_length
                assert len(input_mask) == max_seq_length
                assert len(segment_ids) == max_seq_length

                input_words = example.text_a[:input_len]

                # pad and truncate
                if len(label_ids) > max_seq_length:
                    label_ids = label_ids[:max_seq_length]
                else:
                    pad_length = max_seq_length-len(label_ids)
                    label_ids += [-100]*pad_length
                assert len(label_ids) == max_seq_length

                if entity_id[0] == 50256:
                    negative_sample += 1
                    if tokenizer_name == 'gpt2_for_copy':
                        entity_id = [50257] + entity_id
                    features.append(InputFeatures(input_ids=input_ids, input_mask=input_mask, input_len=input_len,
                                                  segment_ids=segment_ids, input_label=label_ids,
                                                  input_words_ids=[feature_count], input_words=input_words,
                                                  entity=entity_id,
                                                  entity_type=[entity_type_map[example.entity_type]]))
                else:
                    positive_sample += 1
                    if tokenizer_name == 'gpt2_for_copy':
                        entity_id = [50257] + entity_id
                    features.append(InputFeatures(input_ids=input_ids, input_mask=input_mask, input_len=input_len,
                                                  segment_ids=segment_ids, input_label=label_ids,
                                                  input_words_ids=[feature_count], entity=entity_id,
                                                  input_words=input_words,
                                                  entity_type=[entity_type_map[example.entity_type]]))
                feature_count += 1

        elif do_not_separate_type:
            print("**********do not distinguish the type!******************")
            for (ex_index, example) in enumerate(examples):
                if ex_index % 10000 == 0:
                    logger.info("Writing example %d of %d", ex_index, len(examples))
                if type(example.text_a) == list:
                    new_text = ' '.join(example.text_a)
                    tokens = tokenizer.tokenize(' ' + new_text)# 在每句话开头加上空格，保证第一个单词可以被tokenized as G开头
                    sum_length_of_example += len(tokens)
                else:
                    raise(NotImplementedError)
                label_ids = [label_map[x] for x in example.labels]

                # allign the labels for separated words
                new_label = [-100] * len(tokens)
                j = 0
                for i in range(len(tokens)):
                    if 'Ġ' in tokens[i]:
                        new_label[i] = label_ids[j]
                        j = j+1
                    else:
                        new_label[i] = new_label[i-1]
                assert -100 not in new_label# 确保所有label都已成功转换为label ids中的id
                input_ids = tokenizer.convert_tokens_to_ids(tokens)

                # note: input_labels[i] 长度与example.text_a一致，也就是words的长度
                input_labels_dict = get_input_label(example.labels, entity_map, label_map)
                entity_label = get_label_all_type(english, tokenizer, input_labels_dict, type_to_words,
                                                         input_words=example.text_a, label_map=label_map,
                                                         entity_type_map=entity_type_map,  max_len_for_entity=max_len_for_entity,
                                                         sum_entity_length=sum_entity_length, all_entity_frequency=all_entity_frequency,
                                                         type_count=type_count, label_list=label_list, input_ids=input_ids)

                if len(input_ids) > max_seq_length:
                    input_ids = input_ids[: max_seq_length]

                segment_ids = [sequence_a_segment_id] * len(input_ids)
                input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
                input_len = min(len(input_ids), max_seq_length)

                # Zero-pad up to the sequence length.
                padding_length = max_seq_length - len(input_ids)

                input_ids += [pad_token] * padding_length
                input_mask += [0 if mask_padding_with_zero else 1] * padding_length
                segment_ids += [pad_token_segment_id] * padding_length

                assert len(input_ids) == max_seq_length
                assert len(input_mask) == max_seq_length
                assert len(segment_ids) == max_seq_length
                input_words = example.text_a[:input_len]

                if j == len(label_ids):
                    label_ids = label_ids[:max_seq_length]
                    label_ids += [-100] * (max_seq_length - len(label_ids))
                    features.append(InputFeatures(input_ids=input_ids, input_mask=input_mask, input_len=input_len,
                                                  segment_ids=segment_ids, input_label=label_ids,
                                                  input_words_ids=[feature_count], input_words=input_words,
                                                  entity=entity_label,
                                                  entity_type=[0]))
                    feature_count += 1

        else: # the original one!
            for (ex_index, example) in enumerate(examples):
                if ex_index % 10000 == 0:
                    logger.info("Writing example %d of %d", ex_index, len(examples))
                if type(example.text_a) == list:
                    new_text = ' '.join(example.text_a)
                    if lower_case:
                        new_text = new_text.lower()
                        for i in range(len(example.text_a)):
                            example.text_a[i] = example.text_a[i].lower()

                    tokens = tokenizer.tokenize(' ' + new_text)# 在每句话开头加上空格，保证第一个单词可以被tokenized as G开头

                    sum_length_of_example += len(tokens)
                else:
                    raise(NotImplementedError)
                true_label_place = [k for k in range(len(example.labels)) if example.labels[k] != 'O' ]
                true_entity = [example.text_a[k] for k in true_label_place]
                true_entity_set = set(true_entity)
                if len(true_entity_set) != len(true_entity):
                    mismatch += 1

                label_ids = [label_map[x] for x in example.labels]
                # allign the labels for separated words
                new_label = [-100] * len(tokens)
                j = 0
                for i in range(len(tokens)):
                    if 'Ġ' in tokens[i]:
                        new_label[i] = label_ids[j]
                        j = j+1
                    else:
                        new_label[i] = new_label[i-1]
                        # if label_all_tokens:
                        # 如果计算metric时采用entity label id转换回BIO label, 这里是B还是I都可以
                        # 如果直接采用数据集中的BIO label, 需要将B-转换为I-，否则无法与predict entity转化的BIO label匹配
                        # if 'B-' in label_list[new_label[i-1]]:
                        #     new_label[i] = new_label[i-1]+1# 将B-转换成I-
                        # else:
                        #     new_label[i] = new_label[i-1]
                        # else:
                        #     raise(ValueError("do not label off!"))

                assert -100 not in new_label# 确保所有label都已成功转换为label ids中的id
                input_ids = tokenizer.convert_tokens_to_ids(tokens)

                # note: input_labels[i] 长度与example.text_a一致，也就是words的长度
                input_labels_dict = get_input_label(example.labels, entity_map, label_map)
                entity_dict, entity_type_map = get_label(english, tokenizer, input_labels_dict,
                                                          input_words=example.text_a, label_map=label_map,
                                                          entity_map=entity_map,  max_len_for_entity=max_len_for_entity,
                                                          sum_entity_length =sum_entity_length, all_entity_frequency=all_entity_frequency,
                                                          type_count=type_count, label_list=label_list, input_ids=input_ids)

                special_tokens_count = 0
                if len(input_ids) > max_seq_length - special_tokens_count:
                    input_ids = input_ids[: (max_seq_length - special_tokens_count)]

                segment_ids = [sequence_a_segment_id] * len(input_ids)
                input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
                input_len = min(len(input_ids), max_seq_length)

                # Zero-pad up to the sequence length.
                padding_length = max_seq_length - len(input_ids)
                if pad_on_left:
                    input_ids = ([pad_token] * padding_length) + input_ids
                    input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
                    segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
                else:
                    input_ids += [pad_token] * padding_length
                    input_mask += [0 if mask_padding_with_zero else 1] * padding_length
                    segment_ids += [pad_token_segment_id] * padding_length

                assert len(input_ids) == max_seq_length
                assert len(input_mask) == max_seq_length
                assert len(segment_ids) == max_seq_length

                input_words = example.text_a[:input_len]

                # todo 某些数据(空的数据) 仍然无法被正常align, conll中这样的数据有 32 条, ontonote中这样的数据有38条
                if j == len(label_ids):
                    step = 0
                    for i in entity_dict.keys():
                        # pad and truncate
                        if len(input_labels_dict[i]) > max_seq_length:
                            input_labels_dict[i] = input_labels_dict[i][:max_seq_length]
                        else:
                            pad_length = max_seq_length-len(input_labels_dict[i])
                            input_labels_dict[i] += [-100]*pad_length
                        assert len(input_labels_dict[i]) == max_seq_length

                        if entity_dict[i][0] == 50256:
                            negative_sample += 1
                            if tokenizer_name == 'gpt2_for_copy':
                                entity_dict[i] = [50257] + entity_dict[i][:-1]
                            features.append(InputFeatures(input_ids=input_ids, input_mask=input_mask, input_len=input_len,
                                                              segment_ids=segment_ids, input_label=input_labels_dict[i],
                                                              input_words_ids=[feature_count*entity_num+step], input_words=input_words,
                                                              entity=entity_dict[i],
                                                              entity_type=[entity_type_map[i]]))
                        else:
                            positive_sample += 1
                            if tokenizer_name == 'gpt2_for_copy':
                                entity_dict[i] = [50257] + entity_dict[i][:-1]
                            features.append(InputFeatures(input_ids=input_ids, input_mask=input_mask, input_len=input_len,
                                                          segment_ids=segment_ids, input_label=input_labels_dict[i],
                                                          input_words_ids=[feature_count*entity_num+step], entity=entity_dict[i],
                                                          input_words=input_words,
                                                          entity_type=[entity_type_map[i]]))
                        step += 1
                    feature_count += 1

        print("average length in dataset:"+str(sum_length_of_example/ex_index))# 注意ex_index每次load数据时有变化，不要fix
        print_entity_length(sum_entity_length, type_count, entity_type_map)
        print("feature count", feature_count)
        print("mismatch：same entity(words) appear in one sentence *************"+str(mismatch) +"********************")
        return features, count, positive_sample, negative_sample, type_count, extend_token, cur_vocab_id, all_entity_frequency

    else:# 中文
        if dataset == 'ontonote4':
            entity_num = 4
        else:
            raise(NotImplementedError)
        type_count = [0] * entity_num
        sum_entity_length = [0] * entity_num
        feature_count = 0
        print("chinese:only use bert-base-chinese tokenizer")
        for (ex_index, example) in enumerate(examples):
            if ex_index % 10000 == 0:
                logger.info("Writing example %d of %d", ex_index, len(examples))

            input_ids = tokenizer.convert_tokens_to_ids(example.text_a)
            sum_length_of_example += len(input_ids)

            true_label_place = [k for k in range(len(example.labels)) if example.labels[k] != 'O' ]
            true_entity = [example.text_a[k] for k in true_label_place]
            true_entity_set = set(true_entity)
            if len(true_entity_set) != len(true_entity):
                mismatch += 1

            # Account for [CLS] and [SEP] with "- 2".
            special_tokens_count = 0
            if len(input_ids) > max_seq_length - special_tokens_count:
                input_ids = input_ids[: (max_seq_length - special_tokens_count)]
                example.labels = example.labels[: (max_seq_length - special_tokens_count)]
                example.text_a = example.text_a[: (max_seq_length - special_tokens_count)]

            input_len = min(len(input_ids)+special_tokens_count, max_seq_length)
            # cls_token = "[CLS]"
            # sep_token = "[SEP]"
            # cls_token_segment_id = 1
            # input_ids += [tokenizer.convert_tokens_to_ids(sep_token)]
            # example.text_a += [sep_token]
            # example.labels += ['O']
            segment_ids = [sequence_a_segment_id] * len(input_ids)
            # input_ids = [tokenizer.convert_tokens_to_ids(cls_token)] + input_ids
            # example.labels = ['O']+example.labels
            # example.text_a = [cls_token] + example.text_a
            # segment_ids = [cls_token_segment_id] + segment_ids

            input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            # input_ids2 = tokenizer.encode(new_text) # encode会自动加上101 102

            # note: input_labels[i] 长度与example.text_a一致，也就是words的长度
            input_labels_dict = get_input_label(example.labels, entity_map, label_map)
            entity_dict, entity_type_map = get_label(english, tokenizer, input_labels_dict,
                                                     input_words=example.text_a, label_map=label_map,
                                                     entity_map=entity_map,  max_len_for_entity=max_len_for_entity,
                                                     sum_entity_length=sum_entity_length, all_entity_frequency=all_entity_frequency,
                                                     type_count=type_count, label_list=label_list, input_ids=input_ids)
            # Zero-pad up to the sequence length.
            padding_length = max_seq_length - len(input_ids)
            if pad_on_left:
                input_ids = ([pad_token] * padding_length) + input_ids
                input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
                segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            else:
                input_ids += [pad_token] * padding_length
                input_mask += [0 if mask_padding_with_zero else 1] * padding_length
                segment_ids += [pad_token_segment_id] * padding_length

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            step = 0
            for i in entity_dict.keys():
                if len(example.text_a) != len(input_labels_dict[i]):
                    print("wrong")
                pad_length = max_seq_length-len(input_labels_dict[i])
                input_labels_dict[i] += [-100]*pad_length
                assert len(input_labels_dict[i]) == max_seq_length
                if entity_dict[i][0] == tokenizer.convert_tokens_to_ids(tokenizer.tokenize(('无')))[0]:
                    negative_sample += 1
                    # note: input_label=input_labels_dict[i] 和 input_words 都没有加入cls和sep token

                    features.append(InputFeatures(input_ids=input_ids, input_mask=input_mask, input_len=input_len,
                                                  segment_ids=segment_ids, input_label=input_labels_dict[i],
                                                  input_words_ids=[feature_count*entity_num+step], input_words=example.text_a,
                                                  entity=entity_dict[i],
                                                  entity_type=[entity_type_map[i]]))
                else:
                    positive_sample += 1

                    features.append(InputFeatures(input_ids=input_ids, input_mask=input_mask, input_len=input_len,
                                                  segment_ids=segment_ids, input_label=input_labels_dict[i],
                                                  input_words_ids=[feature_count*entity_num+step], entity=entity_dict[i],
                                                  input_words=example.text_a,
                                                  entity_type=[entity_type_map[i]]))
                step += 1
            feature_count += 1
        print("average length in dataset:"+str(sum_length_of_example/ex_index))# 注意ex_index每次load数据时有变化，不要fix
        print_entity_length(sum_entity_length, type_count, entity_type_map)
        print("feature count", feature_count)
        print("mismatch：same entity(words) appear in one sentence *************"+str(mismatch) +"********************")
        return features, count, positive_sample, negative_sample, type_count, extend_token, cur_vocab_id, all_entity_frequency


class Conll2003Processor(DataProcessor):
    """Processor for an english ner data set."""

    def get_train_examples(self, data_dir, limit=None):
        """See base class."""
        return self._create_examples(self._read_text(os.path.join(data_dir, "train.txt")), "train", limit)

    def get_dev_examples(self, data_dir, limit=None):
        """See base class."""
        return self._create_examples(self._read_text(os.path.join(data_dir, "testa.txt")), "dev", limit)

    def get_test_examples(self, data_dir,limit=None):
        """See base class."""
        return self._create_examples(self._read_text(os.path.join(data_dir, "testb.txt")), "test", limit)

    def get_labels(self):
        """See base class."""
        return ['B-PER', 'I-PER', 'B-LOC',  'I-LOC', 'B-MISC', 'I-MISC', 'B-ORG', 'I-ORG', 'O']# "X", "[START]", "[END]"
        # donot change the order!

    def _create_examples(self, lines, set_type, limit=None):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            if limit != None:
                if i > limit:
                    break
            guid = "%s-%s" % (set_type, i)
            text_a = line['words']
            # BIOS
            labels = []
            for x in line['labels']:
                if 'M-' in x:
                    labels.append(x.replace('M-','I-'))
                elif 'E-' in x:
                    labels.append(x.replace('E-', 'I-'))
                else:
                    labels.append(x)
            examples.append(InputExample(guid=guid, text_a=text_a, labels=labels))
        return examples

class OntonoteProcessor(DataProcessor):
    """Processor for an english ner data set."""

    def get_train_examples(self, data_dir, limit=None):
        """See base class."""
        return self._create_examples(self._read_text(os.path.join(data_dir, "train.sd.conllx"), 'ontonote'), "train", limit)

    def get_dev_examples(self, data_dir, limit=None):
        """See base class."""
        return self._create_examples(self._read_text(os.path.join(data_dir, "dev.sd.conllx"), 'ontonote'), "dev", limit)

    def get_test_examples(self, data_dir,limit=None):
        """See base class."""
        return self._create_examples(self._read_text(os.path.join(data_dir, "test.sd.conllx"), 'ontonote'), "test", limit)


    def get_labels(self):
        """See base class."""
        return ['B-NORP', 'I-NORP',
                'B-GPE', 'I-GPE',
                'B-FAC', 'I-FAC',
                'B-PERSON',  'I-PERSON',
                'B-DATE', 'I-DATE',
                'B-ORG', 'I-ORG',
                'B-LOC', 'I-LOC',
                'B-WORK_OF_ART', 'I-WORK_OF_ART',
                'B-CARDINAL', 'I-CARDINAL',
                'B-ORDINAL', 'I-ORDINAL',
                'B-PRODUCT', 'I-PRODUCT',
                'B-QUANTITY', 'I-QUANTITY',
                'B-TIME', 'I-TIME',
                'B-EVENT', 'I-EVENT',
                'B-PERCENT', 'I-PERCENT',
                'B-MONEY', 'I-MONEY',
                'B-LAW', 'I-LAW',
                'B-LANGUAGE', 'I-LANGUAGE',
                'O'] #"X", "[START]", "[END]"
        # donot change the order!

    def _create_examples(self, lines, set_type, limit=None):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            if limit != None:
                if i > limit:
                    break
            guid = "%s-%s" % (set_type, i)
            text_a = line['words']
            # BIOS
            labels = []
            for x in line['labels']:
                if 'M-' in x:
                    labels.append(x.replace('M-', 'I-'))
                elif 'E-' in x:
                    labels.append(x.replace('E-', 'I-'))
                else:
                    labels.append(x)
            examples.append(InputExample(guid=guid, text_a=text_a, labels=labels))
        return examples


class Conll2003MRCProcessor(DataProcessor):
    """Processor for an english ner data set."""

    def get_train_examples(self, data_dir, limit=None):
        """See base class."""
        all_data = self._read_json(os.path.join(data_dir, "mrc-ner.train"))
        labels = self._read_text(os.path.join('datasets/conll2003_bio', "train.txt"))
        lines = []
        for i in range(len(labels)):
            words = all_data[4*i]['context'].split(' ')
            lines.append({"words": words, "labels": labels[i]['labels']})
        return self._create_examples(lines, "train", limit)

    def get_dev_examples(self, data_dir, limit=None):
        """See base class."""
        all_data = self._read_json(os.path.join(data_dir, "mrc-ner.dev"))
        labels = self._read_text(os.path.join('datasets/conll2003_bio', "testa.txt"))
        lines = []
        for i in range(len(labels)):
            words = all_data[4*i]['context'].split(' ')
            lines.append({"words": words, "labels": labels[i]['labels']})
        return self._create_examples(lines, "dev", limit)

    def get_test_examples(self, data_dir, limit=None):
        """See base class."""
        all_data = self._read_json(os.path.join(data_dir, "mrc-ner.test"))
        labels = self._read_text(os.path.join('datasets/conll2003_bio', "testb.txt"))
        lines = []
        for i in range(len(labels)):
            words = all_data[4*i]['context'].split(' ')
            lines.append({"words": words, "labels": labels[i]['labels']})
        return self._create_examples(lines, "test", limit)


    def get_labels(self):
        """See base class."""
        return ['B-PER', 'I-PER', 'B-LOC',  'I-LOC', 'B-MISC', 'I-MISC', 'B-ORG', 'I-ORG', 'O']# "X", "[START]", "[END]"
        # donot change the order!

    def _create_examples(self, lines, set_type, limit=None):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            if limit != None:
                if i > limit:
                    break
            guid = "%s-%s" % (set_type, i)
            text_a = line['words']
            # BIOS
            labels = []
            for x in line['labels']:
                if 'M-' in x:
                    labels.append(x.replace('M-','I-'))
                elif 'E-' in x:
                    labels.append(x.replace('E-', 'I-'))
                else:
                    labels.append(x)
            examples.append(InputExample(guid=guid, text_a=text_a, labels=labels))
        return examples


class GENIAMRCProcessor(DataProcessor):
    """Processor for an english ner data set."""

    def get_train_examples(self, data_dir, limit=None):
        """See base class."""
        lines = self._read_json2(os.path.join(data_dir, "mrc-ner.train"), 'genia')
        return self._create_examples(lines, "train", limit)

    def get_dev_examples(self, data_dir, limit=None):
        """See base class."""
        lines = self._read_json2(os.path.join(data_dir, "mrc-ner.dev"), 'genia')
        return self._create_examples(lines, "dev", limit)

    def get_test_examples(self, data_dir, limit=None):
        """See base class."""
        lines = self._read_json2(os.path.join(data_dir, "mrc-ner.test"),'genia')
        return self._create_examples(lines, "test", limit)


    def get_labels(self):
        """See base class."""
        return ["B-cell_line","I-cell_line", "B-cell_type",  "I-cell_type", "B-DNA", "I-DNA", "B-RNA", "I-RNA",
                "B-protein", "I-protein", 'O']
        # donot change the order!

    def _create_examples(self, lines, set_type, limit=None):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            if limit != None:
                if i > limit:
                    break
            guid = "%s-%s" % (set_type, i)
            text_a = line['words']
            # BIOS
            labels = []
            for x in line['labels']:
                if 'M-' in x:
                    labels.append(x.replace('M-', 'I-'))
                elif 'E-' in x:
                    labels.append(x.replace('E-', 'I-'))
                else:
                    labels.append(x)
            examples.append(InputExample(guid=guid, text_a=text_a, labels=labels, entity_type=line['entity_type']))
        return examples

class ACE05MRCProcessor(DataProcessor):
    """Processor for an english ner data set."""

    def get_train_examples(self, data_dir, limit=None):
        """See base class."""
        lines = self._read_json2(os.path.join(data_dir, "mrc-ner.train"), 'ace05')
        return self._create_examples(lines, "train", limit)

    def get_dev_examples(self, data_dir, limit=None):
        """See base class."""
        lines = self._read_json2(os.path.join(data_dir, "mrc-ner.dev"),'ace05')
        return self._create_examples(lines, "dev", limit)

    def get_test_examples(self, data_dir, limit=None):
        """See base class."""
        lines = self._read_json2(os.path.join(data_dir, "mrc-ner.test"), 'ace05')
        return self._create_examples(lines, "test", limit)

    def get_labels(self):
        """See base class."""
        return ["B-GPE", "I-GPE",
                "B-ORG", "I-ORG",
                "B-PER", "I-PER",
                "B-FAC", "I-FAC",
                "B-VEH", "I-VEH",
                "B-LOC", "I-LOC",
                "B-WEA", "I-WEA",
                'O']
        # donot change the order!

    def _create_examples(self, lines, set_type, limit=None):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            if limit != None:
                if i > limit:
                    break
            guid = "%s-%s" % (set_type, i)
            text_a = line['words']
            # BIOS
            labels = []
            for x in line['labels']:
                if 'M-' in x:
                    labels.append(x.replace('M-', 'I-'))
                elif 'E-' in x:
                    labels.append(x.replace('E-', 'I-'))
                else:
                    labels.append(x)
            examples.append(InputExample(guid=guid, text_a=text_a, labels=labels, entity_type=line['entity_type']))
        return examples

class Ontonote4Processor(DataProcessor):
    """Processor for the chinese ner data set."""

    def get_train_examples(self, data_dir, limit):
        """See base class."""
        return self._create_examples(self._read_text2(os.path.join(data_dir, "train.char.bmes")), "train", limit)

    def get_dev_examples(self, data_dir, limit):
        """See base class."""
        return self._create_examples(self._read_text2(os.path.join(data_dir, "dev.char.bmes")), "dev", limit)

    def get_test_examples(self, data_dir, limit):
        """See base class."""
        return self._create_examples(self._read_text2(os.path.join(data_dir, "test.char.bmes")), "test", limit)

    def get_labels(self, markup='bio'):
        """See base class."""
        return ['B-PER',  'I-PER', 'B-LOC', 'I-LOC', 'B-GPE', 'I-GPE', 'B-ORG', 'I-ORG', "O"]

    def _create_examples(self, lines, set_type, limit=None):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if limit != None:
                if i > limit:
                    break
            guid = "%s-%s" % (set_type, i)
            text_a = line['words']
            # BIOS
            labels = []
            for x in line['labels']:
                x = x.strip('\n')
                # change the labels in cner dataset to BIO style
                if 'M-' in x:
                    labels.append(x.replace('M-', 'I-'))
                elif 'E-' in x:
                    labels.append(x.replace('E-', 'I-'))
                elif 'S-' in x:
                    labels.append(x.replace('S-', 'B-'))
                else:
                    labels.append(x)
            # labels[-1] = 'O'
            examples.append(InputExample(guid=guid, text_a=text_a, labels=labels))
        return examples

ner_processors = {
    'conll2003': Conll2003Processor,
    'ontonote': OntonoteProcessor,
    'ontonote4': Ontonote4Processor,
    'conll2003_mrc': Conll2003MRCProcessor,
    'genia': GENIAMRCProcessor,
    'ace05':ACE05MRCProcessor
}
