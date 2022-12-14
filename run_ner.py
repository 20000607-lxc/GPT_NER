import os
import json
import time
import torch.distributed as dist
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from callback.optimizater.adamw import AdamW
from callback.optimizater.adafactor import AdaFactor
from callback.lr_scheduler import get_linear_schedule_with_warmup
from callback.progressbar import ProgressBar
from common import seed_everything
from common import init_logger, logger
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from models.gpt_for_ner_base import BareGPT2_loop
from models.gpt_for_ner import *
from processors.ner_seq import convert_examples_to_features, collate_fn, write_stat_in_files
from processors.ner_seq import ner_processors as processors
from ner_metrics import NewSeqEntityScore, compute_hit1
from finetuning_argparse import get_argparse
from transformers import AutoTokenizer
import wandb

MODEL_CLASSES = {
    'gpt2_loop': (GPT2Config, GPT2GenerateWithPointer_loop),
    'bare_gpt2_loop': (GPT2Config, BareGPT2_loop),
    'gpt2_copy_loop': (GPT2Config, GPT2GenerateWithPointer_loop_test_copy),
}
TEMPLATE_CLASSES = {
    '1': (6, 8, 0),

}
args = get_argparse().parse_args()
assert args.use_discrete == True
if args.model_type == 'beam_search':
    assert args.tokenizer_name == 'gpt2_for_copy'
    assert args.per_gpu_eval_batch_size == 1
else:
    assert args.per_gpu_eval_batch_size != 1

if args.tokenizer_name == 'gpt2_for_copy':
    assert args.model_type in ['gpt2_copy_loop', 'beam_search'] and args.use_extend_vocab == True

if args.tokenizer_name == 'gpt2_get_all_entity':
    assert args.use_extend_vocab == False

assert args.remove_all_after_end == True
TRAIN_LIMIT = args.train_limit
EVAL_LIMIT = args.eval_limit
TEST_LIMIT = args.test_limit

if args.task_name in ["conll2003", "conll2003_mrc", 'ontonote4']:
    ENTITY_TYPE = 4
elif args.task_name == "ontonote":
    ENTITY_TYPE = 18

elif args.task_name == 'genia':
    ENTITY_TYPE = 5
elif args.task_name == 'ace05':
    ENTITY_TYPE = 7


def train(args, train_dataset, model, tokenizer,extend_token, extend_vocab_id ):
    """ Train the model """
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,
                                  collate_fn=collate_fn)
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]

    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        "weight_decay": args.weight_decay, },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    args.warmup_steps = int(t_total * args.warmup_proportion)
    if args.optimizer_choice == 'adafactor':
        optimizer = AdaFactor(optimizer_grouped_parameters, lr=args.learning_rate)
    else:
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=t_total)
    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
        os.path.join(args.model_name_or_path, "scheduler.pt")):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (dist.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)
    global_step = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if os.path.exists(args.model_name_or_path) and "checkpoint" in args.model_name_or_path:
        # set global_step to global_step of last saved checkpoint from model path
        global_step = int(args.model_name_or_path.split("-")[-1].split("/")[0])
        epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
        steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)
        logger.info("  Continuing training from checkpoint, will skip to saved global_step")
        logger.info("  Continuing training from epoch %d", epochs_trained)
        logger.info("  Continuing training from global step %d", global_step)
        logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    seed_everything(args.seed)  # Added here for reproductibility (even between python 2 and 3)
    for epoch in range(int(args.num_train_epochs)):
        pbar = ProgressBar(n_total=len(train_dataloader), desc='Training')
        for step, batch in enumerate(train_dataloader):
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {"input_ids": batch[0], "attention_mask": batch[1],
                      "entities": batch[4], "entity_type": batch[5]}
            if args.model_type != "distilbert":
                # XLM and RoBERTa don't use segment_ids
                inputs["token_type_ids"] = (batch[2] if args.model_type in ["bert", "xlnet"] else None)
            outputs = model(**inputs)
            loss = outputs[0]
            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            pbar(step, {'loss': loss.item()})

            if args.use_wandb:
                wandb.log({'loss': loss.item()})

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

        logger.info("\n")

        # at the end of each epoch do eval and test !
        evaluate(args, model, tokenizer, extend_token, extend_vocab_id, args.model_type)

        predict(args, model, tokenizer, extend_token, extend_vocab_id, args.model_type)

        if 'cuda' in str(args.device):
            torch.cuda.empty_cache()

    return global_step, tr_loss / global_step

def evaluate(args, model, tokenizer, extend_token, extend_vocab_id, prefix):
    print('************ evaluation *******************')
    metric = NewSeqEntityScore(id2label=args.id2label, markup=args.markup, dataset=args.task_name, tokenizer=tokenizer,
                               remove_all_after_end=args.remove_all_after_end)
    eval_output_dir = os.path.join(args.output_file_dir, prefix)
    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    eval_dataset,  all_input_ids, all_input_words, type_count, extend_token, extend_vocab_id, all_entity_frequency \
        = load_and_cache_examples(args, args.task_name,  tokenizer, data_type='dev', limit=EVAL_LIMIT,
                                   extend_token=extend_token, cur_vocab_id=extend_vocab_id)
    print("************ total extend ids from eval dataset:(should be the same with from train dataset" +
          str(extend_vocab_id) + "***************")

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size,
                                 collate_fn=collate_fn)
    # Eval!
    logger.info("***** Running evaluation %s *****", prefix)
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0

    all_hit1 = [0] * ENTITY_TYPE
    example_num = 0
    pbar = ProgressBar(n_total=len(eval_dataloader), desc="Evaluating")
    print_result = args.print_results
    output_result = []
    if print_result:
        json_d = {}
        json_d["basic"] = args.note
        json_d["Eval"] = "Eval"
        output_result.append(json_d)
    for step, batch in enumerate(eval_dataloader):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            example_num += len(batch[0])
            inputs = {"input_ids": batch[0], "attention_mask": batch[1],
                      "entities": batch[4], "entity_type": batch[5]}
            if args.model_type != "distilbert":
                # XLM and RoBERTa don"t use segment_ids
                inputs["token_type_ids"] = (batch[2] if args.model_type in ["bert", "xlnet"] else None)
            outputs = model(**inputs)
        tmp_eval_loss, hit1,  all_pred_entities, example,  all_true_entities = outputs
        if args.n_gpu > 1:
            tmp_eval_loss = tmp_eval_loss.mean()  # mean() to average on multi-gpu parallel evaluating
        eval_loss += tmp_eval_loss.item()
        nb_eval_steps += 1
        for i in range(len(all_hit1)):
            all_hit1[i] += hit1[i]
        for j in range(len(batch[0])):
            index = step*args.eval_batch_size+j
            assert batch[0][j][0] == all_input_ids[index][0]
            assert batch[7][j] == index
            json_d = metric.update(pred_paths=all_pred_entities.tolist()[j], label_paths=batch[4].tolist()[j],
                          entity_types=batch[5].tolist()[j],
                          input_labels=batch[6].tolist()[j],
                          input_words=all_input_words[index])

            if json_d != {}:
                example_j = example[j].tolist()
                example_j = [i for i in example_j if i !=50257]
                json_d['output'] = tokenizer.decode(example_j)
                output_result.append(json_d)
        pbar(step)

    logger.info("\n")
    eval_loss = eval_loss / nb_eval_steps
    new = True
    if new:
        eval_info = metric.result('eval')
        results = {f'{key}': value for key, value in eval_info.items()}
        results['eval_loss'] = eval_loss
        results["eval_all_hit1"] = sum(all_hit1)
        results["eval_hit1_proportion"] = sum(all_hit1)/example_num
        logger.info("***** Eval results %s *****", prefix)
        info = "-".join([f' {key}: {value:.4f} ' for key, value in results.items()])
        logger.info(info)
    else:
        eval_info, entity_info = metric.result('eval')
        results = {f'{key}': value for key, value in eval_info.items()}
        results['eval_loss'] = eval_loss

        results["eval_all_hit1"] = sum(all_hit1)
        results["eval_hit1 proportion"] = sum(all_hit1)/example_num
        logger.info("***** Eval results %s *****", prefix)
        info = "-".join([f' {key}: {value:.4f} ' for key, value in results.items()])
        logger.info(info)
        logger.info("***** Entity results %s *****", prefix)
        for key in sorted(entity_info.keys()):
            logger.info("******* %s results ********"%key)
            info = "-".join([f' {key}: {value:.4f} ' for key, value in entity_info[key].items()])
            logger.info(info)
    if args.use_wandb:
        wandb.log(results)

    if print_result:
        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)
        output_submit_file = os.path.join(eval_output_dir, args.output_file_name)
        json_d = compute_hit1(all_hit1, type_count, dataset=args.task_name)
        output_result.append(json_d)
        with open(output_submit_file, "w") as writer:
            for record in output_result:
                if args.task_name == 'ontonote4':
                    writer.write(json.dumps(record, ensure_ascii=False) + '\n')
                else:
                    writer.write(json.dumps(record) + '\n')

    return results

def predict(args, model, tokenizer, extend_token, extend_vocab_id, prefix):
    # calculate all the predict wrong types
    predict_wrong_type = {}
    for k in range(len(args.id2label)):
        predict_wrong_type_in = {}
        predict_wrong_type[args.id2label[k]] = predict_wrong_type_in
        for k in range(len(args.id2label)):
            predict_wrong_type_in[args.id2label[k]] = 0
   # calculate all the predict wrong types for rare entities
    rare_predict_wrong_type = {}
    for k in range(len(args.id2label)):
        rare_predict_wrong_type_in = {}
        rare_predict_wrong_type[args.id2label[k]] = rare_predict_wrong_type_in
        for k in range(len(args.id2label)):
            rare_predict_wrong_type_in[args.id2label[k]] = 0

    print('********************* test **************************' )
    metric = NewSeqEntityScore(id2label=args.id2label, markup=args.markup, dataset=args.task_name, tokenizer=tokenizer,
                               remove_all_after_end=args.remove_all_after_end)
    pred_output_dir = os.path.join(args.output_file_dir, prefix)
    if not os.path.exists(pred_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(pred_output_dir)
    test_dataset, all_input_ids, all_input_words, type_count, extend_token, cur_vocab_id, all_entity_frequency\
        = load_and_cache_examples(args, args.task_name, tokenizer=tokenizer, data_type='test',limit=TEST_LIMIT,
                                  extend_token=extend_token, cur_vocab_id=extend_vocab_id)

    # ???????????????entity??????????????? rare_entity = dict{key = entity token, value = frequency}
    data_output_dir = args.output_file_dir
    rare_entity = write_stat_in_files(all_entity_frequency, data_output_dir, 'stat_of_test_data', tokenizer)
    print("write rare entity in output files ! ")
    print("************total extend ids from test dataset:(should be the same with from train dataset" +
          str(extend_vocab_id) + "***************")

    # Note that DistributedSampler samples randomly
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    test_sampler = SequentialSampler(test_dataset) if args.local_rank == -1 else DistributedSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.eval_batch_size, collate_fn=collate_fn)
    # Eval!
    logger.info("***** Running test %s *****", prefix)
    logger.info("  Num examples = %d", len(test_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    output_submit_file = os.path.join(pred_output_dir,  args.output_file_name)
    all_hit1 = [0]*ENTITY_TYPE
    example_num = 0
    print_result = args.print_results
    pbar = ProgressBar(n_total=len(test_dataloader), desc="Predicting")
    output_result = []
    if print_result:
        json_d = {}
        json_d["basic"] = args.note
        json_d["test"] = "test"
        output_result.append(json_d)
    for step, batch in enumerate(test_dataloader):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            example_num += len(batch[0])
            inputs = {"input_ids": batch[0], "attention_mask": batch[1],
                      "entities": batch[4], "entity_type": batch[5],  'test': True}
            if args.model_type != "distilbert":
                # XLM and RoBERTa don"t use segment_ids
                inputs["token_type_ids"] = (batch[2] if args.model_type in ["bert", "xlnet"] else None)
            outputs = model(**inputs)

        hit1, all_pred_entities, example, all_true_entities = outputs
        for i in range(len(all_hit1)):
            all_hit1[i] += hit1[i]# list + list = concat

        for j in range(len(batch[0])):
            index = step*args.eval_batch_size + j
            assert batch[0][j][0] == all_input_ids[index][0]
            assert batch[7][j] == index
            json_d = metric.update(pred_paths=all_pred_entities.tolist()[j], label_paths=batch[4].tolist()[j],
                          entity_types=batch[5].tolist()[j], pred_wrong_type=predict_wrong_type,
                          rare_pred_wrong_type=rare_predict_wrong_type, rare_entity=rare_entity,
                          input_labels=batch[6].tolist()[j],
                          input_words=all_input_words[index])

            if json_d != {}:
                example_j = example[j].tolist()
                example_j = [i for i in example_j if i !=50257]
                json_d['output'] = tokenizer.decode(example_j)
                output_result.append(json_d)
        # if print_result:
        #     ensemble_examples(args.task_name, example, all_true_entities, step, tokenizer,  batch[0].tolist(),
        #                       output_result, batch[5].tolist(), args.TEMPLATE)
        pbar(step)
    logger.info("\n")
    new = True
    if new:
        test_info = metric.result()
        results = {f'{key}': value for key, value in test_info.items()}
        results["all_hit1"] = sum(all_hit1)
        results["hit1 proportion"] = sum(all_hit1)/example_num
        logger.info("***** Test results %s *****", prefix)
        info = "-".join([f' {key}: {value:.4f} ' for key, value in results.items()])
        logger.info(info)
    else:
        eval_info, entity_info = metric.result()
        results = {f'{key}': value for key, value in eval_info.items()}
        results["all_hit1"] = sum(all_hit1)
        results["hit1 proportion"] = sum(all_hit1)/example_num
        logger.info("***** Test results %s *****", prefix)
        info = "-".join([f' {key}: {value:.4f} ' for key, value in results.items()])
        logger.info(info)
        logger.info("***** Test Entity results %s *****", prefix)
        for key in sorted(entity_info.keys()):
            logger.info("******* %s results ********"%key)
            info = "-".join([f' {key}: {value:.4f} ' for key, value in entity_info[key].items()])
            logger.info(info)

    json_d = {}
    json_d['predict_wrong_type'] = "the following is the predict wrong types "
    output_result.append(json_d)
    for i in predict_wrong_type.keys():
        json_d = {}
        json_d[i] = predict_wrong_type[i]
        output_result.append(json_d)

    json_d = {}
    json_d['rare_predict_wrong_type'] = "the following is the rare predict wrong types "
    output_result.append(json_d)
    for i in rare_predict_wrong_type.keys():
        json_d = {}
        json_d[i] = rare_predict_wrong_type[i]
        output_result.append(json_d)

    for k in range(len(metric.wrong_labels)):
        if k % 3 == 0:
            json_d = {}
            json_d['input words'] = metric.wrong_labels[k]
            json_d['input labels'] = metric.wrong_labels[k+2]
            json_d['entity label converted label'] = metric.wrong_labels[k+1]
        output_result.append(json_d)

    json_d = {}
    json_d['upper_case_number'] = metric.upper_case
    json_d['upper_case_wrong'] = metric.upper_case_wrong
    output_result.append(json_d)

    if print_result:
        with open(output_submit_file, "w") as writer:
            for record in output_result:
                if args.task_name == 'ontonote4':
                    writer.write(json.dumps(record, ensure_ascii=False) + '\n')
                else:
                    writer.write(json.dumps(record) + '\n')
        print("write the test results in files: " + str(output_submit_file))
    if args.use_wandb:
        wandb.log(results)

def load_and_cache_examples(args, task, tokenizer, data_type='train', limit=None, extend_token=None, cur_vocab_id=None):
    if args.local_rank not in [-1, 0] and not evaluate:
        dist.barrier()
        # Make sure only the first process in distributed training process the dataset, and the others will use the cache\

    processor = processors[task]()
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(args.data_dir, 'cached_soft-{}_{}_{}_{}'.format(
        data_type,
        list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.train_max_seq_length if data_type=='train' else args.eval_max_seq_length),
        str(task)))
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        raise(NotImplementedError)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        assert label_list[0] != 'O'# ??????label_list?????????????????????????????????

        if data_type == 'train':
            examples = processor.get_train_examples(args.data_dir, limit)
            # examples2 = processor.get_dev_examples(args.data_dir, limit)
            # examples.extend(examples2)

        elif data_type == 'dev':
            examples = processor.get_dev_examples(args.data_dir, limit)
        else:
            examples = processor.get_test_examples(args.data_dir, limit)

        if args.task_name in [ 'ontonote4']:
            ENGLISH = False
        else:
            ENGLISH = True

        # gpt2tokenizer ??????sep_token  pad_token cls_token ????????????None
        features, count, positive, negative, type_count, extend_token, cur_vocab_id, all_entity_frequency \
            = convert_examples_to_features(english=ENGLISH, lower_case=args.do_lower_case,
                                        dataset=args.task_name,
                                        task_name=data_type,
                                        max_len_for_entity=args.max_len,
                                        tokenizer_name=args.tokenizer_name,
                                        examples=examples,
                                        tokenizer=tokenizer,
                                        label_list=label_list,
                                        max_seq_length=args.train_max_seq_length if data_type=='train'\
                                                else args.eval_max_seq_length,
                                        pad_on_left=False,
                                        # pad on the left for xlnet
                                        pad_token=0,
                                        pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0,
                                           extend_token=extend_token, cur_vocab_id=cur_vocab_id)
        print("number of examples whose labels cannot be aligned "+str(count))# ???????????????????????????examples???token_ids ?????????0???
        print("positive samples: " + str(positive))
        print("negative samples: " + str(negative))
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)
    if args.local_rank == 0 and not evaluate:
        dist.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache
    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_lens = torch.tensor([f.input_len for f in features], dtype=torch.long)
    all_entity_labels = torch.tensor([f.entity for f in features], dtype=torch.long)
    all_entity_types = torch.tensor([f.entity_type for f in features], dtype=torch.long)
    all_input_labels = torch.tensor([f.input_label for f in features], dtype=torch.long)
    all_match_input_ids = torch.tensor([f.input_words_ids for f in features], dtype=torch.long)

    if data_type != 'train':
        # they are list not tensor for input words are list or string not integers!
        all_input_words = [f.input_words for f in features]
    else:
        all_input_words = 0
    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                            all_lens, all_entity_labels, all_entity_types, all_input_labels, all_match_input_ids )
    return dataset, all_input_ids, all_input_words, type_count, extend_token, cur_vocab_id, all_entity_frequency

def main():
    args = get_argparse().parse_args()
    if args.task_name in ['ontonote4']:
        ENGLISH = False
    else:
        ENGLISH = True
    args.project = 'entity generation NEW ' + args.task_name
    args.local_rank = -1
    if args.use_wandb:
        wandb.init(config=args, project=args.project, entity='entity name', group=args.group_name)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    args.output_dir = args.output_dir + '{}'.format(args.model_type)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    time_ = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
    init_logger(log_file=args.output_dir + f'/{args.model_type}-{args.task_name}-{time_}.log')
    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir))
    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda", args.cuda)
        # torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 1# torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '38281'
        dist.init_process_group(backend="nccl", rank=0, world_size=1)# dist = torch.distributed
        args.n_gpu = 1

    args.device = device
    logger.warning(
                "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16,)
    # Set seed
    seed_everything(args.seed)
    # Prepare NER task
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name]()
    label_list = processor.get_labels()
    args.id2label = {i: label for i, label in enumerate(label_list)}
    args.label2id = {label: i for i, label in enumerate(label_list)}
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        dist.barrier()  # Make sure only the first process in distributed training will download model & vocab
    args.model_type = args.model_type.lower()
    config_class, model_class = MODEL_CLASSES[args.model_type]
    TEMPLATE = TEMPLATE_CLASSES[args.template]
    args.TEMPLATE = TEMPLATE
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          num_labels=num_labels,
                                          loss_type=args.loss_type,
                                          cache_dir=args.cache_dir if args.cache_dir else None,)
    if 'gpt2' in args.model_name_or_path:
        # if args.task_name == 'ontonote4':
        #     from transformers import BertTokenizer
        #     tokenizer = BertTokenizer.from_pretrained("uer/gpt2-chinese-cluecorpussmall")
        # else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path if ENGLISH else 'bert-base-chinese', use_fast=False)# use_fast=True, add_prefix_space=True
        model = model_class(config=config, device=args.device, template=TEMPLATE,
                            use_discrete=args.use_discrete,  fine_tune=args.fine_tune,
                            dataset=args.task_name, assume_true_length=args.assume_true_length,
                            use_extend_vocab=args.use_extend_vocab if args.model_type in ['gpt2_copy_loop', 'beam_search'] else None,
                            model_name=args.model_name_or_path, generated_pipeline=args.generated_pipeline,
                            filling_value=args.filling_value)
    else:
        raise(NotImplementedError)

    if args.local_rank == 0:
        dist.barrier() # Make sure only the first process in distributed training will download model & vocab
    model.to(args.device)
    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        data_output_dir = args.output_file_dir
        train_dataset, _, _, type_count, extend_token, extend_vocab_id, all_entity_frequency \
            = load_and_cache_examples(args, task=args.task_name, tokenizer=tokenizer,
                                      data_type='train', limit=TRAIN_LIMIT, extend_token={}, cur_vocab_id=50257)
        # ???????????????entity???????????????
        write_stat_in_files(all_entity_frequency, data_output_dir, 'stat_of_train_data', tokenizer)

        print("************total extend ids from train dataset:" + str(extend_vocab_id) + "*****************")
        global_step, tr_loss = train(args, train_dataset, model, tokenizer, extend_token, extend_vocab_id)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1 or dist.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)
        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`

        # model_to_save = (
        #     model.module if hasattr(model, "module") else model
        # )  # Take care of distributed/parallel training
        # model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_vocabulary(args.output_dir)# should save vocabulary !
        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))
        # logger.info("Saving model checkpoint to %s", args.output_dir)
        # torch.save(model.state_dict(), os.path.join(args.output_dir, "model.pt"))



if __name__ == "__main__":
    main()