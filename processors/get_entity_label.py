
def get_label_all_type(english, tokenizer, input_labels_dict, type_to_words, input_words, label_map, entity_type_map,
                       max_len_for_entity, sum_entity_length, all_entity_frequency, type_count, label_list, input_ids):
    end_token_id = 50256
    sep_token_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(('*')))[0]
    type_sep_token_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(('$')))[0]
    sep_token = '*'
    type_sep_token = ' $'
    join_token = ' :'
    entity_label_dict = {}
    for i in entity_type_map.keys():
        entity_label_dict[i] = []

    for i in input_labels_dict.keys():
        already_have_this_type = 0
        for j in range(len(input_labels_dict[i])):
            if input_labels_dict[i][j] != label_map['O']:
                # if j != 0 and 'B-' in label_list[input_labels_dict[i][j]] and 'B-' in label_list[input_labels_dict[i][j-1]]:
                # 两个B（也就是两个同种类的entity) ，采用分隔符构成固定pattern
                if already_have_this_type and 'B-' in label_list[input_labels_dict[i][j]]:
                    entity_label_dict[i].append(sep_token)
                already_have_this_type = 1
                entity_label_dict[i].append(input_words[j])# word level

    entity_label = []
    for i in entity_label_dict.keys():
        len_entity = len(entity_label_dict[i])

        if len_entity > max_len_for_entity:
            entity_label_dict[i] = entity_label_dict[i][:max_len_for_entity]

        # 统计entity种类以及长度
        if len_entity != 0:
            type_count[entity_type_map[i]] += 1
            sum_entity_length[entity_type_map[i]] += len_entity
            # entity_label_i = type_to_words[i] + join_token + ' '.join(entity_label_dict[i]) + type_sep_token# person: Trump $
            entity_label_i = type_to_words[i] + ' ' + ' '.join(entity_label_dict[i]) + type_sep_token# person Trump $
            entity_label_i = tokenizer.tokenize(' '+entity_label_i)
            entity_label_i_removed = [j for j in entity_label_i if 'Ġ' in j]# 注意sep token * 也要保留啊！
            entity_label_id_removed = tokenizer.convert_tokens_to_ids(entity_label_i_removed)

            entity_label.extend(entity_label_id_removed)

    if entity_label != []:
        entity_label[-1] = end_token_id
    else:
        entity_label = [end_token_id]

    entity_label = entity_label[:max_len_for_entity]
    entity_label = entity_label + (max_len_for_entity-len(entity_label))*[-100]


    return entity_label



def get_input_label(labels, entity_map, label_map):
    # 初始化
    return_label_dict = {}
    j = 0
    for i in entity_map.values():
        if j % 2 == 0:
            return_label_dict[i] = [label_map['O']]*len(labels)
        j = j+1

    # 按照entity种类取对应的BI tag
    for i in range(len(labels)):
        if labels[i] != 'O':
            return_label_dict[labels[i][2:]][i] = label_map[labels[i]]# [2:]指去掉B-或者I-
    return return_label_dict


def get_label(english, tokenizer, input_labels_dict, input_words, label_map, entity_map,  max_len_for_entity, sum_entity_length,
              all_entity_frequency, type_count, label_list, input_ids):
    entity_type_map = {}
    end_token_id = 50256 if english else tokenizer.convert_tokens_to_ids(tokenizer.tokenize(('无')))[0]# note: 102 is the id of sep-token for bert-base-chinese tokenizer
    # dict used to record entities: {key: entity_name, value: id for this entity type, in range(num_entity_types)}
    j = 0
    for i in entity_map.values():
        if j % 2 == 0:
            entity_type_map[i] = j//2
        j = j+1

    sep_token_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(('*')))[0] if english else tokenizer.convert_tokens_to_ids(tokenizer.tokenize(('*')))[0]
    # todo : this token cna be ' *', ',' etc.
    # conll可以用11， 一句正常的话逗号前面是不会有空格的， 所以只能指定id为11（不能tokenize(',')， 会得到837，不是11)
    # genia不可以用11, 因为genia数据集有的entity中本身含有逗号， 因此只能用*

    na_list = [-100] * max_len_for_entity
    entity_label_dict = {}
    j = 0
    for i in entity_map.values():
        if j % 2 == 0:
            entity_label_dict[i] = []
        j = j+1
    for i in input_labels_dict.keys():
        already_have_this_type = 0
        for j in range(len(input_labels_dict[i])):
            if input_labels_dict[i][j] != label_map['O']:
                # if j != 0 and 'B-' in label_list[input_labels_dict[i][j]] and 'B-' in label_list[input_labels_dict[i][j-1]]:
                    # 两个B（也就是两个同种类的entity) ，采用分隔符构成固定pattern
                if already_have_this_type and 'B-' in label_list[input_labels_dict[i][j]]:
                    entity_label_dict[i].append(sep_token_id)
                already_have_this_type = 1

                if english:
                    # 英语单词存在被分为tokens的
                    tokens_j = tokenizer.tokenize(' '+input_words[j])
                    tokens_j = tokenizer.convert_tokens_to_ids(tokens_j)
                    # 验证分两步和encode效果一致
                    # token_p = tokenizer.encode(' '+input_words[j])
                    # if token_p != tokens_j:
                    #     print(tokens_j)
                    # for token in tokens_j:
                    #     assert token in input_ids
                        # entity_label_dict[i].append(token) #  1 BPE level

                    entity_label_dict[i].append(tokens_j[0])

                else:
                    entity_label_dict[i].append(input_ids[j])

    for i in entity_label_dict.keys():
        len_entity = len(entity_label_dict[i])
        # 统计entity出现频率
        entity_token = tokenizer.decode(entity_label_dict[i])
        if entity_token in all_entity_frequency.keys():
            all_entity_frequency[entity_token][0] += 1
            if all_entity_frequency[entity_token][0] > 3:
                # 如果frequency 大于 3， 不再存储相应的input tokens
                all_entity_frequency[entity_token] = [all_entity_frequency[entity_token][0]]
        else:
            all_entity_frequency[entity_token] = [1]
            all_entity_frequency[entity_token].append(input_words)

        # append end_token_id
        if len_entity > (max_len_for_entity-1):
            entity_label_dict[i] = entity_label_dict[i][:max_len_for_entity-1]
            entity_label_dict[i].append(end_token_id)
        elif 0 < len_entity <= (max_len_for_entity-1):
            entity_label_dict[i].append(end_token_id)
            entity_label_dict[i].extend((max_len_for_entity-1 - len_entity)*[-100])
        elif len_entity == 0:
            entity_label_dict[i].extend(max_len_for_entity*[-100])

        # 统计entity种类以及长度
        if len_entity != 0:
            type_count[entity_type_map[i]] += 1
            sum_entity_length[entity_type_map[i]] += len_entity
        else:
            entity_label_dict[i] = na_list
            entity_label_dict[i][0] = end_token_id
    return entity_label_dict, entity_type_map


def get_label_nested(tokenizer, input_labels, input_words, max_len_for_entity, sum_entity_length,
              all_entity_frequency, type_count, input_ids, entity_type_map, entity_type):

    end_token_id = 50256
    sep_token_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(('*')))[0] # 一句正常的话逗号前面是不会有空格的， 所以只能指定id为11（不能tokenize(',')， 会得到837，不是11)

    na_list = [-100] * max_len_for_entity
    entity_label = []

    already_have_this_type = 0
    for j in range(len(input_labels)):
        if input_labels[j] != 'O':
            # if j != 0 and 'B-' in input_labels_dict[i][j] and 'B-' in input_labels_dict[i][j-1]:
            # 两个B（也就是两个同种类的entity) ，采用分隔符构成固定pattern
            if already_have_this_type and 'B-' in input_labels[j]:
                entity_label.append(sep_token_id)
            already_have_this_type = 1
            # 英语单词存在被分为tokens的
            tokens_j = tokenizer.tokenize(' '+input_words[j])
            tokens_j = tokenizer.convert_tokens_to_ids(tokens_j)
            # 验证分两步和encode效果一致
            # token_p = tokenizer.encode(' '+input_words[j])
            # if token_p != tokens_j:
            #     print(tokens_j)
            # for token in tokens_j:
            #     assert token in input_ids
                # entity_label.append(token) # 1 for BPE level
            entity_label.append(tokens_j[0])

    len_entity = len(entity_label)
    # 统计entity出现频率
    entity_token = tokenizer.decode(entity_label)
    if entity_token in all_entity_frequency.keys():
        all_entity_frequency[entity_token][0] += 1
        if all_entity_frequency[entity_token][0] > 3:
            # 如果frequency 大于 3， 不再存储相应的input tokens
            all_entity_frequency[entity_token] = [all_entity_frequency[entity_token][0]]
    else:
        all_entity_frequency[entity_token] = [1]
        all_entity_frequency[entity_token].append(input_words)

    # append end_token_id
    if len_entity > (max_len_for_entity-1):
        entity_label = entity_label[:max_len_for_entity-1]
        entity_label.append(end_token_id)
    elif 0 < len_entity <= (max_len_for_entity-1):
        entity_label.append(end_token_id)
        entity_label.extend((max_len_for_entity-1 - len_entity)*[-100])
    elif len_entity == 0:
        entity_label.extend(max_len_for_entity*[-100])

    # 统计entity种类以及长度
    if len_entity != 0:
        type_count[entity_type_map[entity_type]] += 1
        sum_entity_length[entity_type_map[entity_type]] += len_entity
    else:
        entity_label = na_list
        entity_label[0] = end_token_id
    return entity_label



# def get_all_entities_label(tokenizer, type_count, new_label, input_ids, max_len_for_entity, sum_entity_length,
#                            entity_map, all_entity_frequency):
#     """
#     return all the entities and construct negative samples for all the entity types that are not in the original sequence
#     Args:
#         new_label: BIO tags for the input
#         input_ids: input sequence ids
#         max_len_for_entity: 8 or 16, set in args
#         sum_entity_length:list:[entity_types] record the length of each entity type
#         entity_map: dict of entity_types(contain "no entity")
#     Returns:
#         entity_type, entity, entity_type_wrong, entity_wrong, entity_type_map, return_new_label
#     """
#     sep_token = tokenizer.encode('*')
#     entity_type_map = {}# dict used to record entities: {key: entity_name, value: id for this entity type, in range(num_entity_types)}
#     entity_type_map["no entity"] = 0
#     j = 0
#     for i in entity_map.values():
#         if j % 2 == 0:
#             entity_type_map[i] = j//2
#         j = j+1
#
#     labels = {}# dict to record entities: {key:entity name, value: 对应input中该entity的id}
#     j = 0
#     for i in entity_map.values():
#         if j % 2 == 0:
#             labels[i] = []
#         j = j+1
#     labels["no entity"] = []
#
#     for i in range(len(new_label)):
#         if new_label[i] >= 0 and new_label[i] < len(entity_map):# means skip -100(ignored token)  and label 'O' whose id is equal to len(entity_map)
#             labels[entity_map[new_label[i]]].append(input_ids[i])
#
#     na_list = [-100] * max_len_for_entity
#     # pad or truncate to max_len
#     for i in labels.keys():
#         len_entity = len(labels[i])
#         # 统计entity出现频率
#         entity_token = tokenizer.decode(labels[i])
#         if entity_token in all_entity_frequency.keys():
#             all_entity_frequency[entity_token][0] += 1
#             if all_entity_frequency[entity_token][0] > 3:
#                 # 如果frequency 大于 3， 不再存储相应的input tokens
#                 all_entity_frequency[entity_token] = [all_entity_frequency[entity_token][0]]
#         else:
#             all_entity_frequency[entity_token] = [1]
#             all_entity_frequency[entity_token].append(tokenizer.decode(input_ids))
#
#         # append 50256
#         if len_entity > (max_len_for_entity-1):
#             labels[i] = labels[i][:max_len_for_entity-1]
#             labels[i].append(50256)
#         elif 0 < len_entity <= (max_len_for_entity-1):
#             labels[i].append(50256)
#             labels[i].extend((max_len_for_entity-1 - len_entity)*[-100])
#         elif len_entity == 0:
#             labels[i].extend(max_len_for_entity*[-100])
#
#         # 统计entity种类以及长度
#         if len_entity != 0:
#             type_count[entity_type_map[i]] += 1
#             sum_entity_length[entity_type_map[i]] += len_entity
#         else:
#             labels[i] = na_list
#             labels[i][0] = 50256
#     return labels, entity_type_map

def print_entity_length(sum_entity_length, type_count, entity_type_map):
    key = list(entity_type_map.keys())
    for i in range(len(sum_entity_length)):
        c = sum_entity_length[i]/type_count[i] if type_count[i] != 0 else 0
        print("average {} length in dataset:".format(key[i]) + str(c))