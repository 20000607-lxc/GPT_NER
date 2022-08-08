from seqeval.metrics import f1_score, precision_score, accuracy_score, recall_score, classification_report

def find_all_index(arr,item):
    return [i for i, a in enumerate(arr) if a == item]

def getmaxstr(str1, str2):
    lstr1 = len(str1)
    lstr2 = len(str2)
    record = [[0 for i in range(lstr2+1)] for j in range(lstr1+1)]
    maxNum = 0
    p = 0
    for i in range(lstr1):
        for j in range(lstr2):
            if str1[i] == str2[j]:
                record[i+1][j+1] = record[i][j]+1
                if record[i+1][j+1] > maxNum:
                    maxNum = record[i+1][j+1]
                    p = i+1
    return p-maxNum, p-1

def get_labels_bio(true_labels):
    if len(true_labels) == 0:
        return true_labels

    if len(true_labels) == 1:
        if true_labels[0] != 'O' and 'B-' not in true_labels[0]:
            true_labels[0] = 'B-' + true_labels[0]
        return true_labels

    for j in range(1, len(true_labels)-1):

        if true_labels[j] != 'O' and true_labels[j-1] == 'O' and 'B-' not in true_labels[j]:
            true_labels[j] = 'B-' + true_labels[j]
        elif true_labels[j] != 'O' and true_labels[j-1] != 'O' and 'B-' not in true_labels[j]:
            true_labels[j] = 'I-' + true_labels[j]

    if true_labels[0] != 'O' and 'B-' not in true_labels[0]:
        true_labels[0] = 'B-' + true_labels[0]

    if true_labels[-1] != 'O' and 'B-' not in true_labels[-1]:
        if true_labels[-2] != 'O':
            true_labels[-1] = 'I-' + true_labels[-1]
        else:
            true_labels[-1] = 'B-' + true_labels[-1]
    return true_labels

def allign_words(words, english):
    if english:
        new_words = []
        for j in range(len(words)):
            if words[j] != '':
                if '\'s' not in words[j]:
                    if ',' in words[j]:

                        new_words.append(words[j].strip(','))
                        new_words.append(',')
                    elif '\'' in words[j]:

                        split_word = words[j].split('\'')
                        new_words.append(split_word[0])
                        new_words.append('\'')
                        new_words.append(split_word[1])
                    else:
                        new_words.append(words[j])
                else:
                    new_words.append(words[j].strip('\'s'))
                    new_words.append('\'s')
    else:
        new_words = words
    return new_words


def get_label_from_entities(english, entities, input_words, entity_types,  entity_type):
    """
    for 区分prompt type 的模型
    每次只get一种entity type的BIO tag
    """
    new_input_words = [input_words[i] for i in range(len(input_words))]
    true_labels = ['O'] * len(input_words)
    if entities != ['']:
        for j in range(len(entities)):
            entity_j = entities[j].split(' ')
            entity_j = allign_words(entity_j, english)
            start, end = getmaxstr(new_input_words, entity_j)
            if start == end:
                true_labels[start] = 'B-' + entity_type[entity_types[0]]
            elif end == -1:# 没有共同的字符
                continue
            else:
                true_labels[start] = 'B-' + entity_type[entity_types[0]]
                true_labels[start+1:end+1] = [entity_type[entity_types[0]]]*(end-start)
            new_input_words[start:end+1] = ['[UNK]']*(end-start+1)
    return true_labels


def get_all_label_from_entities(english, all_entities, input_words, word2label, this_is_label):
    """
    entities = [person Trump, location USA*England, organization EU]
    """
    true_labels = ['O'] * len(input_words)
    new_input_words = [input_words[i] for i in range(len(input_words))]

    for i in range(len(all_entities)):
        all_entities[i] = all_entities[i].split(' ')# location,  USA*England
        if this_is_label:
            assert all_entities[i][0] == ''

        if len(all_entities[i]) <= 2:
            continue
        entity_type = all_entities[i][1]
        entities = ' '.join(all_entities[i][2:])

        entities = entities.split('*')# USA, England
        if entities != [''] and entity_type in word2label.keys():
            for j in range(len(entities)):
                entity_j = entities[j].split(' ')
                entity_j = allign_words(entity_j, english)
                start, end = getmaxstr(new_input_words, entity_j)
                if start == end:
                    true_labels[start] = 'B-' + word2label[entity_type]
                elif end == -1:# 没有共同的字符
                    continue
                else:
                    true_labels[start] = 'B-' + word2label[entity_type]
                    true_labels[start+1:end+1] = [word2label[entity_type]]*(end-start)
                new_input_words[start:end+1] = ['[UNK]']*(end-start+1)
    return true_labels


def write_in_json(json_d, pred_labels, new_input_true_labels, entity_type, entity_types,
                  english, pred_words, label_words, input_words,
                  lower_case_label_str, label_str, upper_case_wrong):
    for i in range(len(pred_labels)):
        if pred_labels[i] != new_input_true_labels[i]:
            json_d['entity_type'] = entity_type[entity_types[0]]
            json_d['predict entity'] = pred_words if pred_words != [] else ''
            json_d['real_entity'] = label_words
            json_d['input'] = input_words
            json_d['pred_labels'] = ' '.join(pred_labels)
            json_d['true_labels'] = ' '.join(new_input_true_labels)
            if lower_case_label_str != label_str:
                upper_case_wrong += 1
            break
    return json_d


class NewSeqEntityScore(object):
    def __init__(self,  id2label, tokenizer, markup='bio', dataset='conll2003', remove_all_after_end=True):
        self.markup = markup
        self.remove_all_after_end = remove_all_after_end
        self.id2label = id2label
        self.tokenizer = tokenizer
        self.reset()
        self.count_forBB = 0
        self.end_token_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(('无')))[0] if dataset == 'ontonote4' else 50256
        self.vocab_size = 21128 if dataset == 'ontonote4' else 50257
        self.english = True if dataset != 'ontonote4' else False
        self.sep_token = '*' if dataset == 'ontonote4' else '*'
        self.type_sep_token = '$'

        self.do_not_separate_type = True

        self.not_same_length_count = 0
        if dataset in ['conll2003', 'conll2003_mrc']:
            self.entity_type = {
                0: "PER",
                1: "LOC",
                2: "MISC",
                3: "ORG"
            }
            self.word2label = {
                'person': 'PER',
                'location': 'LOC',
                'others': 'MISC',
                'organization': 'ORG'
            }
        elif dataset == 'ontonote':
            self.entity_type = {
                0: 'NORP',
                1: 'GPE',
                2: 'FAC',
                3: 'PERSON',
                4: 'DATE',
                5: 'ORG',
                6: 'LOC',
                7: 'WORK_OF_ART',
                8: 'EVENT',
                9: 'CARDINAL',
                10: 'ORDINAL',
                11: "PRODUCT",
                12: 'QUANTITY',
                13: 'TIME',
                14: 'PERCENT',
                15: 'MONEY',
                16: 'LAW',
                17: 'LANGUAGE'
            }
            self.word2label = {
                'nationality': 'NORP',
                'government': 'GPE',
                'buildings': 'FAC',
                'person': 'PERSON',
                'date': 'DATE',
                'organization': 'ORG',
                'location': 'LOC',
                'work_of_art': 'WORK_OF_ART',
                'event': 'EVENT',
                'cardinal': 'CARDINAL',
                'ordinal': 'ORDINAL',
                'product': "PRODUCT",
                'quantity': 'QUANTITY',
                'time': 'TIME',
                'percent': 'PERCENT',
                'money': 'MONEY',
                'law': 'LAW',
                'language': 'LANGUAGE'
            }

        elif dataset == 'ontonote4':
            self.entity_type = {
                0: "PER",
                1: "LOC",
                2: "GPE",
                3: "ORG"
            }
        elif dataset == 'genia':
            self.entity_type = {
                0:'cell_line',
                1:'cell_type',
                2:'DNA',
                3:'RNA',
                4:'protein'
            }

    def reset(self):
        self.origins = []
        self.right_all = []
        self.founds = []
        self.wrong_labels = []
        self.converted_origin = []
        self.wrong_num = 0
        self.upper_case_wrong = 0
        self.upper_case = 0

    def result(self, type=None):
        precision = precision_score(self.origins, self.founds)
        accuracy = accuracy_score(self.origins, self.founds)
        recall = recall_score(self.origins, self.founds)
        f1 = f1_score(self.origins, self.founds)
        f1_for_converted = f1_score(self.converted_origin, self.founds)
        print(classification_report(self.origins, self.founds))
        print('\n')
        print("all the converted wrong labels **********"+str(self.wrong_num)+"*****************")
        print('\n')
        if type == 'eval':
            return {'eval_precision': precision, 'eval_recall': recall, 'eval_f1': f1, 'eval_acc': accuracy, 'eval_converted_f1':f1_for_converted}
        else:
            return {'precision': precision, 'recall': recall, 'f1': f1, 'acc': accuracy, 'converted_f1':f1_for_converted}

    def update(self, pred_paths, label_paths, entity_types, pred_wrong_type=None,
               rare_pred_wrong_type=None, rare_entity=None, input_words=None, input_labels=None):

        input_words = self.tokenizer.tokenize(' ' + ' '.join(input_words))
        input_words = [i.strip('Ġ') for i in input_words if 'Ġ' in i]
        json_d = {}
        pred_labels = ['O'] * len(input_words)
        input_labels = [i for i in input_labels if i != -100]
        new_input_true_labels = [self.id2label[k] for k in input_labels]
        assert len(input_words) == len(input_labels)
        self.origins.extend([new_input_true_labels])

        # step1: get the label words
        label_paths = [k for k in label_paths if k != -100 and k < self.vocab_size and k != self.end_token_id]
        #  去掉bos token（50257）和 50256 (endoftext)
        label_str = self.tokenizer.decode(label_paths)
        lower_case_label_str = label_str.lower()
        if lower_case_label_str != label_str:
            self.upper_case += 1

        # step 2: get the prediction words
        pred_paths = [k for k in pred_paths if k != -100 and k < self.vocab_size] #  去掉bos token（50257）
        new_pred_paths = []
        if len(pred_paths) == 0:
            pred_str = ''
        elif pred_paths[0] == self.end_token_id:
            pred_str = ''
        else:
            if self.remove_all_after_end:
                for k in range(len(pred_paths)):
                    if pred_paths[k] == self.end_token_id:
                        break
                    else:
                        new_pred_paths.append(pred_paths[k])
            else:
                for k in range(len(pred_paths)):
                    if pred_paths[k] != self.end_token_id:
                        new_pred_paths.append(pred_paths[k])

            pred_str = self.tokenizer.decode(new_pred_paths)
            pred_words = pred_str.split(self.sep_token)# 转换成['entity1', 'entity2']的形式
            pred_labels = get_label_from_entities(self.english, pred_words, input_words, entity_types, self.entity_type)


        if self.do_not_separate_type:# person Trump $ location USA $ ornazation EU
            label_words = label_str.split(self.type_sep_token)# [person Trump, location USA, organization EU]
            true_labels = get_all_label_from_entities(self.english, label_words, input_words, self.word2label, this_is_label=True)
            true_labels = get_labels_bio(true_labels)
            self.converted_origin.extend([true_labels])

            pred_words = pred_str.split(self.type_sep_token)# 转换成['entity1', 'entity2']的形式
            pred_labels = get_all_label_from_entities(self.english, pred_words, input_words, self.word2label, this_is_label=False)
            pred_labels = get_labels_bio(pred_labels)
            self.founds.extend([pred_labels])

        else:
            label_words = label_str.split(self.sep_token)# 转换成['entity1', 'entity2']的形式
            true_labels = get_label_from_entities(self.english, label_words, input_words, entity_types, self.entity_type)
            # label_words = label_str.split(' ')# 转换成list
            # new_label_words = allign_words(label_words, self.english)
            # true_labels = get_label_from_words(new_label_words, input_words, entity_types, self.sep_token, self.count_forBB, self.entity_type, self.english )
            true_labels = get_labels_bio(true_labels)
            self.converted_origin.extend([true_labels])
            pred_labels = get_labels_bio(pred_labels)
            self.founds.extend([pred_labels])

        # count the converted wrong labels
        if true_labels != new_input_true_labels:
            self.wrong_labels.append(input_words)
            self.wrong_labels.append([true_labels])
            self.wrong_labels.append([new_input_true_labels])
            self.wrong_num += 1

        # step 3: record the pred types for test
        if pred_wrong_type is not None:
            for i in range(len(input_labels)):
                if input_labels[i] != pred_labels[i]:
                    pred_wrong_type[new_input_true_labels[i]][pred_labels[i]] += 1
                    if label_str in rare_entity.keys():
                        rare_pred_wrong_type[new_input_true_labels[i]][pred_labels[i]] += 1

        # step4: write the results in json
        json_d = write_in_json(json_d, pred_labels, new_input_true_labels, self.entity_type, entity_types,
                               self.english, pred_str, label_words, input_words,
                               lower_case_label_str, label_str, self.upper_case_wrong)
        return json_d


def compute_hit1(all_hit1, type_count, dataset):
    json_d = {}
    if dataset == "conll2003:":
        json_d['negative_all_hit1'] = str(all_hit1[0])
        json_d['negative_hit_prop'] = str(all_hit1[0]/type_count[0])
        json_d['per_hit1'] = str(all_hit1[1])
        json_d['per_hit_prop'] = str(all_hit1[1]/type_count[1])
        json_d['loc_hit1'] = str(all_hit1[2])
        json_d['loc_hit_prop'] = str(all_hit1[2]/type_count[2])
        json_d['misc_hit1'] = str(all_hit1[3])
        json_d['misc_hit_prop'] = str(all_hit1[3]/type_count[3])
        json_d['org_hit1'] = str(all_hit1[4])
        json_d['org_hit_prop'] = str(all_hit1[4]/type_count[4])
    elif dataset == 'ontonote':
        pass
    return json_d

