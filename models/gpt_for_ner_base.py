import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.nn.utils.rnn import pad_sequence
from transformers import GPT2LMHeadModel
from transformers import AutoTokenizer

def compute_hit1(num_entity_types, bz, entities, max_len, assume_true_length, generated, entity_type):
    hit1 = [0] * num_entity_types
    pred_right_ids = []
    for i in range(bz):
        use_label = [entities[i][j].item() for j in range(max_len) if entities[i][j].item() != -100]# 用于进行比较的label = [id1, id2]
        # use_label = [ids] or [50256] which means "no"
        if len(use_label) == 0:
            continue
        # elif len(use_label) < assume_true_length:
        #     assume_true_length = len(use_label)

        b = use_label[:assume_true_length]
        c = generated[i][:assume_true_length].tolist()
        # f = generated[i][1:assume_true_length+1].tolist()
        if b == c:
            hit1[entity_type[i]] += 1
            pred_right_ids.append(generated[i][:assume_true_length+1])

    true_entities = entities.tolist()
    return hit1, generated, true_entities

class BareGPT2_loop(torch.nn.Module):
    """
    support discrete prompt only
    used to test different design of questions
    循环生成entity 共循环max_len_of_entity次
    """
    def __init__(self, config, device, template, use_discrete, fine_tune,
                 dataset,assume_true_length=None, use_extend_vocab=None):
        super().__init__()
        self.num_labels = config.num_labels
        self.device = device
        self.gpt = GPT2LMHeadModel.from_pretrained('gpt2').to(self.device)
        self.embeddings = self.gpt.base_model.get_input_embeddings().to(self.device)
        self.dataset = dataset
        if self.dataset == 'conll2003':
            self.num_entity_types = 4
        elif self.dataset == 'ontonote':
            self.num_entity_types = 18
        print(" used dataset name: " + str(self.dataset))

        self.fine_tune = fine_tune
        if not self.fine_tune:
            for param in self.gpt.parameters():
                param.requires_grad = False

        self.assume_true_length = assume_true_length if assume_true_length else 1
        self.dropout = nn.Dropout(config.resid_pdrop).to(self.device)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels).to(self.device)
        self.loss_type = 'ce'
        # prompt setting
        self.hidden_size = self.embeddings.embedding_dim
        self.pad_token_id = 0
        self.pseudo_token_id = 50257
        self.use_discrete = True
        if self.use_discrete:
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
            self.get_discrete_prompt()
            print("init discrete prompt...")
        print("****************  init the right BareGPT2_loop  **********************")
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
                      'language'
                      ]
        if self.dataset == 'ontonote':
            # questions = [" Is there *'any'* entity in this sentence?"]
            questions_entity = [" What is the *'{}'* in this sentence?".format(i) for i in label_list]
            questions =  questions_entity
            self.discrete_prompt = [self.tokenizer.encode(questions[i]) for i in range(self.num_entity_types)]

        elif self.dataset == 'conll2003':
            questions = [
                         " What is the *person* in this sentence?",
                         " What is the *location* in this sentence?",
                         " What is the *event* in this sentence?",
                         " What is the *organization* in this sentence?",
                         ]
            # 顺序与ner_seq中一一对应，discrete prompt的设计加入了高亮符号**（参考PET）
            self.discrete_prompt = [self.tokenizer.encode(questions[i]) for i in range(self.num_entity_types)]

        else:
            raise(ValueError("this dataset is not implemented yet"))

    def get_query(self, input_id, prompt_tokens, entity_type=None):# entity_type=0 1 2 3 4
        input = []
        count = 0
        for i in range(len(input_id)):
            if input_id[i] != 0:
                count += 1
                input.append(input_id[i].item())
        if not self.use_discrete:
            query = input
        else:
            query = input + self.discrete_prompt[entity_type]
            count = count+len(self.discrete_prompt[entity_type])
        return query, count

    def embed_input(self, queries):
        """
        turn the queries(word index) :[batch_size,query_length]
        into embeddings: [batch_size,query_length,768]
        """
        queries_for_embedding = queries.clone()
        queries_for_embedding[(queries == self.pseudo_token_id)] = self.pseudo_token_id-1
        raw_embeds = self.embeddings(queries_for_embedding)
        return raw_embeds

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None,
                head_mask=None, labels=None, entities=None, entity_type=None, extend_ids=None, extend_count=None, test=False):
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

        inputs_embeds = self.embed_input(queries)
        inputs_embeds = inputs_embeds.to(self.device)
        max_len = len(entities[0])

        example = torch.zeros(2, max_len+len(queries[0]))# all tokens from one sentence, choose batch[0]
        all_logits = torch.zeros(bz, max_len, 50257).to(self.device)
        output = self.gpt(inputs_embeds=inputs_embeds, attention_mask=attention_mask1.half())
        logit = output.logits[..., -1, :]# [batch_size, 50257]

        example_logit = torch.argsort(output.logits, dim=2, descending=True)
        example[0, 0:len(queries[0])] = example_logit[0, :, 0]
        generated = torch.zeros_like(entities)# [batch_size, max_len]

        all_logits[:, 0, :] = logit
        token = torch.argsort(logit, dim=1, descending=True)

        context = token[:, 0].unsqueeze(1)# [batch_size, 1]
        past_key_values = output.past_key_values

        for i in range(bz):
            generated[i][0] = context[i].item()

        for ids in range(max_len-1):
            output = self.gpt(context, past_key_values=past_key_values)
            logit = output.logits[..., -1, :]# [batch_size, 50257]

            all_logits[:, ids+1, :] = logit
            token = torch.argsort(logit, dim=1, descending=True)

            context = token[:, 0].unsqueeze(1)# [batch_size, 1]
            past_key_values = output.past_key_values

            for j in range(bz):
                generated[j][ids+1] = context[j].item()

        example[0, len(queries[0]):] = generated[0, :]
        # compute the number and ids of right predictions
        hit1, pred_entity_ids,  true_entities = compute_hit1(self.num_entity_types, bz, entities, max_len, assume_true_length, generated, entity_type)
        example[1, 0:len(input_ids[0])] = input_ids[0]
        # all_logits = torch.softmax(all_logits, dim=2)# [batch_size, entity_length, vocab_size]
        # do not add softmax before CrossEntropyLoss or the loss will always be around 10
        if not test:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(all_logits.view(-1, all_logits.size(-1)), entities.view(-1))
            return loss, hit1, pred_entity_ids, example, true_entities
        else:
            return hit1, pred_entity_ids, example, true_entities



