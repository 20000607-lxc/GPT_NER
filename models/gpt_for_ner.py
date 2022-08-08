import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.nn.utils.rnn import pad_sequence
from transformers import GPT2LMHeadModel
from transformers import AutoTokenizer
from torch.autograd import Variable
import copy
from models.compute_copy.layers import Decoder

def compute_hit1(vocab_size,num_entity_types, bz, entities, max_len, assume_true_length, generated, entity_type):
    hit1 = [0] * num_entity_types
    pred_right_ids = []
    for i in range(bz):
        use_label = [entities[i][j].item() for j in range(max_len) if entities[i][j].item() != -100]
        if len(use_label) == 0:
            continue
        if use_label[0] == vocab_size:
            b = use_label[1:assume_true_length+1]
            c = generated[i][1:assume_true_length+1].tolist()
        else:
            b = use_label[:assume_true_length]
            c = generated[i][:assume_true_length].tolist()
        if b == c:
            hit1[entity_type[i]] += 1
            pred_right_ids.append(generated[i][:assume_true_length+1])
    true_entities = entities.tolist()
    return hit1, true_entities


class GPT2GenerateWithPointer_loop(torch.nn.Module):
    """
    support continuous and discrete prompt
    循环生成entity 共循环max_len_of_entity次
    support conll2003, ontonote, ontonote4.0, genia dataset
    """
    def __init__(self, config, device, template,  use_discrete,  fine_tune, dataset, assume_true_length=None,
                 use_extend_vocab=None, model_name=None, generated_pipeline=None, filling_value=None):
        super().__init__()
        if model_name == None:
            model_name = 'gpt2'

        if dataset == 'ontonote4':
            model_name = "uer/gpt2-chinese-cluecorpussmall"

        self.device = device
        self.gpt = GPT2LMHeadModel.from_pretrained(model_name).to(self.device)
        self.embeddings = self.gpt.base_model.get_input_embeddings().to(self.device)

        self.dataset = dataset
        if self.dataset in ['conll2003', 'conll2003_mrc', 'ontonote4']:
            self.num_entity_types = 4
        elif self.dataset == 'ontonote':
            self.num_entity_types = 18
        elif self.dataset == 'genia':
            self.num_entity_types = 5
        elif self.task_name == 'ace05':
            self.num_entity_types = 7

        print(" used dataset name: " + str(self.dataset))

        self.fine_tune = fine_tune
        if not self.fine_tune:
            for param in self.gpt.parameters():
                param.requires_grad = False

        self.loss_type = 'ce'
        self.vocab_size = 21128 if self.dataset == 'ontonote4' else 50257# prompt word 的id
        self.end_token_id = 102 if self.dataset == 'ontonote4' else 50256
        self.pad_token_id = 0
        self.assume_true_length = assume_true_length if assume_true_length else 1

        # prompt setting
        self.use_discrete = True
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese") if self.dataset=='ontonote4' else AutoTokenizer.from_pretrained("gpt2")
        self.get_discrete_prompt()
        print("init discrete prompt...")

        print("****************  init the right GPT2GenerateWithPointer_loop  **********************")
        print("****************  generate entity in loop  **********************")

    def get_discrete_prompt(self):
        label_list = ['nationalities or religious or political groups ',
                      'Countries, Cities and states',
                      'Buildings,airports, highways, bridges,etc ',
                      'people,(including fictional)',
                      'date',
                      'organization',
                      'location',
                      'titles of books or sounds',
                      'event',
                      'cardinal number',
                      'ordinal number',
                      'products',
                      'quantity',
                      'time',
                      'percentage',
                      'money(including unit)',
                      'law',
                      'language']
        if self.dataset == 'ontonote':
            questions_entity = [" What is the *'{}'* in this sentence? The {} in this sentence is".format(i, i) for i in label_list]
            questions = questions_entity
            self.discrete_prompt = [self.tokenizer.encode(questions[i]) for i in range(self.num_entity_types)]

        elif self.dataset in ['conll2003', 'conll2003_mrc']:
            questions = [
                " Find all the *'person'* entities in this sentence.  The *'person'* entities in this sentence are",
                " Find all the *'location'* entities in this sentence. The *'location'* entities in this sentence are",
                " Find all the *'event'* entities in this sentence. The *'event'* entities in this sentence are",
                " Find all the *'organization'* entities in this sentence. The  *'organization'* entities in this sentence are"]

            desriptions = [
                '*persons* entities are named persons or family.',
                '*places* entities are the name of politically or geographically defined locations such as cities, provinces, countries, international regions, bodies of water, mountains, etc.',
                "*examples of miscellaneous entities* include events, nationalities, products and works of art.",
                "*organization* entities are limited to named corporate, governmental, or other organizational entities."
            ]

            # 顺序与ner_seq中一一对应，discrete prompt的设计加入了高亮符号**（参考PET）
            self.discrete_prompt = [self.tokenizer.encode(questions[i]) for i in range(self.num_entity_types)]
            self.descriptions = [self.tokenizer.encode(desriptions[i]) for i in range(self.num_entity_types)]
        elif self.dataset == 'ontonote4':
            questions = [
                '$这句话中的*人物*都有什么？这句话中的*人物*有：',
                '$这句话中的*地理位置*都有什么？这句话中的*地理位置*有：',
                '$这句话中的*城市、州、国家*都有什么？这句话中的*城市、州、国家*有：',
                '$这句话中的*公司、组织、机构*都有什么？这句话中的*公司、组织、机构*有：',
            ]
            desriptions = [
                '*人物*是个人、家庭的名称。',
                '*地理位置*是城市、省份、国家、自然景观的名称。',
                "*城市、州、国家*只限于人为建立的实体。",
                "*公司、组织、机构*只包含指定的公司、政府或其他组织实体。 "
            ]

            self.discrete_prompt = [self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(questions[i]))   for i in range(self.num_entity_types)]
            self.descriptions    = [self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(desriptions[i])) for i in range(self.num_entity_types)]

        elif self.dataset == 'genia':
            label_list = [
                'cell_line',
                'cell_type',
                'DNA',
                'RNA',
                'protein'
            ]
            desriptions = [
                'cell_line is equal to cell line',
                'cell_type is equal to cell type',
                "DNA",
                "RNA",
                'protein'

            ]
            questions_entity = [" What is the *'{}'* in this sentence? The {} in this sentence is".format(i, i) for i in label_list]
            questions = questions_entity
            self.descriptions    = [self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(desriptions[i])) for i in range(self.num_entity_types)]
            self.discrete_prompt = [self.tokenizer.encode(questions[i]) for i in range(self.num_entity_types)]

        elif self.dataset == 'ace05':
            label_list = [
                'geographical political entities',
                'organization',
                'person',
                'facility',
                'vehicle',
                'location',
                'weapon'
            ]
            desriptions = [
                "*geographical political entities* are geographical regions defined by political and or social groups such as countries, nations, regions, cities, states, government and its people."
                "*organization* entities are limited to companies, corporations, agencies, institutions and other groups of people.",
                "a *person* entity is limited to human including a single individual or a group."
                "*facility* entities are limited to buildings and other permanent man-made structures such as buildings, airports, highways, bridges.",
                "*vehicle* entities are physical devices primarily designed to move, carry, pull or push the transported object such as helicopters, trains, ship and motorcycles.",
                "*location* entities are limited to geographical entities such as geographical areas and landmasses, mountains, bodies of water, and geological formations.",
                "*weapon* entities are limited to physical devices such as instruments for physically harming such as guns, arms and gunpowder.",
            ]
            questions_entity = [" What is the *'{}'* in this sentence? The {} in this sentence is".format(i, i) for i in label_list]
            questions = questions_entity
            self.descriptions    = [self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(desriptions[i])) for i in range(self.num_entity_types)]
            self.discrete_prompt = [self.tokenizer.encode(questions[i]) for i in range(self.num_entity_types)]

        else:
            raise(ValueError("this dataset is not implemented yet"))

    def get_query(self, input_id, prompt_tokens, entity_type=None):
        input = []
        count = 0
        for i in range(len(input_id)):
            if input_id[i] != 0:
                count += 1
                input.append(input_id[i].item())

        if self.dataset=='ontonote4':
            # 不用description  可能是description没写好？
            query = input + self.discrete_prompt[entity_type]
            count = count+len(self.discrete_prompt[entity_type])
        else:
            query = self.descriptions[entity_type] + input + self.discrete_prompt[entity_type]
            count = count+len(self.discrete_prompt[entity_type])+len(self.descriptions[entity_type])

        return query, count

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                head_mask=None, entities=None, entity_type=None, test = False):
        """
        Args:
            input_ids: padded seuqence:[batch_size, max_length]
            attention_mask: [batch_size, max_length]
            token_type_ids: [batch_size, max_length]
            (position_ids: [batch_size, max_length], head_mask: [batch_size, max_length], labels: [batch_size, max_length])
            entities:[batch_size, max_len=8]
            entity_type:[batch_size,1] and the numbers in entity_type are from [0,1,2,3,4]
            extend_ids: padded seuqence:[batch_size, max_length] and the tokens beyond gpt2 vocab also have corresponding ids(>50257) in extend_ids
            extend_count: [batch_size, 1] the number of ids beyond gpt2 vocab in each input_id
        Returns:
            outputs
        """
        assume_true_length = self.assume_true_length
        bz = len(input_ids)
        counts = []
        queries = []
        for i in range(bz):
            query, count = self.get_query(input_ids[i], None, entity_type[i])
            counts.append(count)
            queries.append(torch.LongTensor(query).squeeze(0))

        queries = pad_sequence(queries, True, padding_value=self.pad_token_id).long().to(self.device)
        attention_mask1 = queries != self.pad_token_id

        max_len = len(entities[0])
        example = torch.zeros(bz, max_len+len(queries[0]))# all tokens from one sentence, choose batch[0]
        all_logits = torch.zeros(bz, max_len, self.vocab_size).to(self.device)
        output = self.gpt(input_ids=queries, attention_mask=attention_mask1.half())
        logit = output.logits[..., -1, :]# [batch_size, 50257]

        example_logit = torch.argsort(output.logits, dim=2, descending=True)
        example[:, :len(queries[0])] = example_logit[:, :, 0]
        generated = torch.zeros_like(entities)# [batch_size, max_len]

        all_logits[:, 0, :] = logit
        token = torch.argsort(logit, dim=1, descending=True)
        context = token[:, 0].unsqueeze(1)# [batch_size, 1]
        past_key_values = output.past_key_values
        generated[:, 0] = token[:, 0]

        for ids in range(max_len-1):
            # add teacher forcing when doing gpt2 generate!
            if self.training:
                # train with teacher forcing!
                teacher_context = copy.deepcopy(entities[:, ids]).unsqueeze(1)
                teacher_context[teacher_context < 0] = self.end_token_id
                output = self.gpt(teacher_context, past_key_values=past_key_values)
            else:
                output = self.gpt(context, past_key_values=past_key_values)

            logit = output.logits[..., -1, :]# [batch_size, 50257]
            all_logits[:, ids+1, :] = logit
            token = torch.argsort(logit, dim=1, descending=True)
            context = token[:, 0].unsqueeze(1)# [batch_size, 1]
            past_key_values = output.past_key_values
            generated[:, ids+1] = token[:, 0]

        example[:, len(queries[0]):] = generated
        # compute the number and ids of right predictions
        hit1, all_true_entities = compute_hit1(self.vocab_size, self.num_entity_types, bz, entities, max_len, assume_true_length, generated, entity_type)
        if not test:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(all_logits.view(-1, all_logits.size(-1)), entities.view(-1))
            return loss, hit1, generated, example, all_true_entities
        else:
            return hit1, generated, example, all_true_entities


class GPT2GenerateWithPointer_loop_test_copy(torch.nn.Module):
    """
    use copy , add extend vocab by resize_token_embeddings()
    循环生成entity 共循环max_len_of_entity次
    use the copy mechanism from Pointer net,  do not use the encoder and decoder, only use the attention mechanism
    but we do not use the extend vocab for the hit1 result only from gpt2 is good enough, we expect to use the first token generated by gpt2 to
    performance teacher force for the prediction of following ids.
    only support conll2003, ontonote!
    """
    def __init__(self, config, device, template, use_discrete, fine_tune,
                 dataset, assume_true_length=None, use_extend_vocab=None, model_name=None, generated_pipeline=None):
        super().__init__()
        self.device = device
        self.num_labels = config.num_labels
        self.gpt = GPT2LMHeadModel.from_pretrained(model_name).to(self.device)
        self.extend_number = 0
        self.use_extend_vocab = use_extend_vocab
        if self.use_extend_vocab:
            self.extend_number = 1# only for bos token, whose id is 50257 (assigned in ner_seq.py)
            # if use extend vocab, extend_number is the number of all the OOV words
            self.gpt.resize_token_embeddings(1+50257)

        self.embeddings = self.gpt.base_model.get_input_embeddings().to(self.device)
        self.dataset = dataset
        if self.dataset == 'conll2003':
            self.num_entity_types = 4
        elif self.dataset == 'ontonote':
            self.num_entity_types = 18
        print(" used dataset name: "+ str(self.dataset))

        self.fine_tune = fine_tune
        if not self.fine_tune:
            for param in self.gpt.parameters():
                param.requires_grad = False

        self.dropout = nn.Dropout(config.resid_pdrop).to(self.device)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels).to(self.device)
        self.linear = nn.Linear(2*config.hidden_size, config.hidden_size).to(self.device)
        self.loss_type = 'ce'
        self.pseudo_token_id = self.extend_number+50257# prompt word 的id

        # prompt setting
        self.hidden_size = self.embeddings.embedding_dim
        self.pad_token_id = 0
        self.use_discrete = use_discrete
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")

        self.get_discrete_prompt()
        print("init discrete prompt...")

        # copy setting
        self.model_hidden_dim = self.hidden_size * 2
        self.reduce_dim = nn.Linear(self.hidden_size, self.model_hidden_dim).to(self.device)

        self.decoder = Decoder(emb_dim=self.hidden_size, hidden_dim=self.model_hidden_dim, device=self.device).to(self.device)
        self.assume_true_length = assume_true_length if assume_true_length else 1

        print("****************  init the right GPT2GenerateWithPointer_loop_test_copy  **********************")
        print("****************  generate entity in loop and use teacher force at the second step of generation  **********************")

    def get_discrete_prompt(self):
        label_list = ['nationalities or religious or political groups ',
                      'Countries, Cities and states',
                      'Buildings,airports, highways, bridges,etc ',
                      'people,(including fictional)',
                      'date',
                      'organization',
                      'location',
                      'titles of books or sounds',
                      'event',
                      'cardinal number',
                      'ordinal number',
                      'products',
                      'quantity',
                      'time',
                      'percentage',
                      'money(including unit)',
                      'law',
                      'language']
        if self.dataset == 'ontonote':
            questions_entity = [" What is the *'{}'* in this sentence? The {} in this sentence is".format(i, i) for i in label_list]
            questions = questions_entity
            self.discrete_prompt = [self.tokenizer.encode(questions[i]) for i in range(self.num_entity_types)]

        elif self.dataset == 'conll2003':
            questions = [
                " Find all the *'person'* entities in this sentence?  The *'person'* entities in this sentence is",
                " Find all the *'location'* entities in this sentence? The *'location'* entities in this sentence is",
                " Find all the *'event'* entities in this sentence? The *'event'* entities in this sentence is",
                " Find all the *'organization'* entities in this sentence? The  *'organization'* entities in this sentence is"]


            desriptions = [
                '*person* entities are named persons or family.',
                '*location* entities are the name of politically or geographically defined locations such as cities, provinces, countries, international regions, bodies of water, mountains, etc.',
                "*examples of miscellaneous entities* include events, nationalities, products and works of art.",
                "*organization* entities are limited to named corporate, governmental, or other organizational entities."
            ]
            self.discrete_prompt = [self.tokenizer.encode(questions[i]) for i in range(self.num_entity_types)]
            self.descriptions = [self.tokenizer.encode(desriptions[i]) for i in range(self.num_entity_types)]
        else:
            raise(ValueError("this dataset is not implemented yet"))

    def get_query(self, input_id, entity_type=None):
        input = []
        count = 0
        for i in range(len(input_id)):
            if input_id[i] != 0:
                count += 1
                input.append(input_id[i].item())
        query = self.descriptions[entity_type] + input + self.discrete_prompt[entity_type]
        count = count+len(self.discrete_prompt[entity_type])+len(self.descriptions[entity_type])
        return query, count

    def compute_copy_logit_every_step(self, enc_out,  s_t, enc_padding_mask, c_t, dec_batch, vocab_dist, di, extend_ids):
        """
        compute copy logit each time after gpt2 generate *one logit*
        Args:
            enc_out:embedding of input_ids [batch_size, sequence_length, 768]
            s_t: tuple 2: the average of input ids' embedding : ([batch_size,  768], [batch_size, 768])
            enc_padding_mask: attention_mask [batch_size, sequence_length]
            dec_batch: entity_label [batch_size]
            vocab_dist: gpt2 output logit at step di [batch_size, vocab_size]
            di: int step di
        Returns:
            output_logits: [batch_size, vocab_size]
        """
        if di == 0:
            if self.use_extend_vocab:
                y_t = copy.deepcopy(dec_batch).to(self.device)
                y_t_emb = self.embeddings(y_t)
                no_teacher_force = y_t < 0
            else:
                y_t_emb = None
                no_teacher_force = None
        else:
            y_t = copy.deepcopy(dec_batch).to(self.device)# when change y_t, should not change dec_batch(entities)
            no_teacher_force = y_t < 0
            y_t[y_t < 0] = 50256
            y_t_emb = self.embeddings(y_t)

        final_dist, s_t, c_t, attn_dist, p_gen, next_coverage = \
            self.decoder(y_t_emb, no_teacher_force, s_t, enc_out, enc_padding_mask, c_t, di, vocab_dist,
                               enc_batch_extend_vocab=extend_ids)
        return final_dist, s_t, c_t

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None,
                head_mask=None, labels=None, entities=None, entity_type=None,
                test = False):
        """
        Args:
            input_ids: （comtain the extended ids! ) padded seuqence:[batch_size, max_length]
            attention_mask: [batch_size, max_length]
            token_type_ids: [batch_size, max_length]
            (position_ids: [batch_size, max_length], head_mask: [batch_size, max_length], labels: [batch_size, max_length])
            entities:[batch_size, max_len=8]
            entity_type:[batch_size,1] and the numbers in entity_type are from [0,1,2,3,4]
        Returns:
            outputs
        """
        assume_true_length = self.assume_true_length
        bz = len(input_ids)
        counts = []
        queries = []
        for i in range(bz):
            query, count = self.get_query(input_ids[i],  entity_type[i])
            counts.append(count)
            queries.append(torch.LongTensor(query).squeeze(0))

        queries = pad_sequence(queries, True, padding_value=self.pad_token_id).long().to(self.device)
        attention_mask1 = queries != self.pad_token_id
        max_len = len(entities[0])

        example = torch.zeros(bz, max_len+len(queries[0]))# all tokens from one sentence, choose batch[0]
        all_logits = torch.zeros(bz, max_len, 50257+self.extend_number).to(self.device)
        output = self.gpt(input_ids=queries, attention_mask=attention_mask1.half())
        logit = output.logits[..., -1, :]# [batch_size, 50257]

        example_logit = torch.argsort(output.logits, dim=2, descending=True)
        example[:, :len(queries[0])] = example_logit[:, :, 0]
        generated = torch.zeros_like(entities)# [batch_size, max_len]

        # use copy: init s_t for each batch
        enc_out = self.embeddings(input_ids)
        s_t_1 = self.reduce_dim(enc_out.mean(axis=1).unsqueeze(0))# 2*768
        s_t = (s_t_1, s_t_1)

        c_t = Variable(torch.zeros((bz, 2 * self.model_hidden_dim))).to(self.device)
        _, s_t, c_t = self.compute_copy_logit_every_step(enc_out, s_t, attention_mask, c_t, entities[:, 0], logit, 0, input_ids)

        # not use copy mechanism to change the gpt2 generated logit at first step
        all_logits[:, 0, :] = logit
        token = torch.argsort(logit, dim=1, descending=True)

        context = token[:, 0].unsqueeze(1)# [batch_size, 1]
        past_key_values = output.past_key_values
        generated[:, 0] = token[:, 0]

        for ids in range(max_len-1):
            # add teacher forcing when doing gpt2 generate!
            if self.training:
                # train with teacher forcing!
                teacher_context = copy.deepcopy(entities[:, ids]).unsqueeze(1)
                teacher_context[teacher_context < 0] = 50256
                output = self.gpt(teacher_context, past_key_values=past_key_values)
            else:
                output = self.gpt(context, past_key_values=past_key_values)

            logit = output.logits[..., -1, :]# [batch_size, 50257]

            if not test:
                logit, s_t, c_t = self.compute_copy_logit_every_step(enc_out, s_t, attention_mask, c_t,
                                                                     entities[:, ids], logit, ids+1,input_ids)
                # from step 2, change the logit by copy attention
            else:
                logit, s_t, c_t = self.compute_copy_logit_every_step(enc_out, s_t, attention_mask, c_t,
                                                                     context.squeeze(1), logit, ids+1, input_ids)
                # when test, use the context（gpt2生成的上一个token）

            all_logits[:, ids+1, :] = logit
            token = torch.argsort(logit, dim=1, descending=True)
            context = token[:, 0].unsqueeze(1)# [batch_size, 1]

            past_key_values = output.past_key_values
            generated[:, ids+1] = token[:, 0]

        example[:, len(queries[0]):] = generated
        # compute the number and ids of right predictions
        hit1, all_true_entities = compute_hit1(self.num_entity_types, bz, entities, max_len, assume_true_length, generated, entity_type)
        if not test:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(all_logits.view(-1, all_logits.size(-1)), entities.view(-1))
            return loss, hit1, generated,  example, all_true_entities
        else:
            return hit1, generated,  example, all_true_entities
