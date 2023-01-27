import argparse
import json
import os
from collections import OrderedDict
import torch
import csv
import util
from transformers import DistilBertTokenizerFast
from transformers import DistilBertForQuestionAnswering
from transformers import AdamW
from tensorboardX import SummaryWriter

import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from args import get_train_test_args
from pathlib import Path

from tqdm import tqdm

def prepare_eval_data(dataset_dict, tokenizer):
    tokenized_examples = tokenizer(dataset_dict['question'],
                                   dataset_dict['context'],
                                   truncation="only_second",
                                   stride=128,
                                   max_length=384,
                                   return_overflowing_tokens=True,
                                   return_offsets_mapping=True,
                                   padding='max_length')
    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

    # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
    # corresponding example_id and we will store the offset mappings.
    tokenized_examples["id"] = []
    tokenized_examples["data_set_id"] = []
    for i in tqdm(range(len(tokenized_examples["input_ids"]))):
        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)
        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        tokenized_examples["id"].append(dataset_dict["id"][sample_index])
        tokenized_examples["data_set_id"].append(dataset_dict["data_set_id"][sample_index])
        # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
        # position is part of the context or not.
        tokenized_examples["offset_mapping"][i] = [
            (o if sequence_ids[k] == 1 else None)
            for k, o in enumerate(tokenized_examples["offset_mapping"][i])
        ]

    return tokenized_examples



def prepare_train_data(dataset_dict, tokenizer):
    print(dataset_dict.keys())
    tokenized_examples = tokenizer(dataset_dict['question'],
                                   dataset_dict['context'],
                                   truncation="only_second",
                                   stride=128,
                                   max_length=384,
                                   return_overflowing_tokens=True,
                                   return_offsets_mapping=True,
                                   padding='max_length')
    sample_mapping = tokenized_examples["overflow_to_sample_mapping"]
    offset_mapping = tokenized_examples["offset_mapping"]

    # Let's label those examples!
    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []
    tokenized_examples['id'] = []
    tokenized_examples['data_set_id'] = []
    inaccurate = 0
    for i, offsets in enumerate(tqdm(offset_mapping)):
        # We will label impossible answers with the index of the CLS token.
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        answer = dataset_dict['answer'][sample_index]
        # Start/end character index of the answer in the text.
        start_char = answer['answer_start'][0]
        end_char = start_char + len(answer['text'][0])
        tokenized_examples['id'].append(dataset_dict['id'][sample_index])
        tokenized_examples['data_set_id'].append(dataset_dict['data_set_id'][sample_index])
        # Start token index of the current span in the text.
        token_start_index = 0
        while sequence_ids[token_start_index] != 1:
            token_start_index += 1

        # End token index of the current span in the text.
        token_end_index = len(input_ids) - 1
        while sequence_ids[token_end_index] != 1:
            token_end_index -= 1

        # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
        if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
            # Note: we could go after the last offset if the answer is the last word (edge case).
            while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                token_start_index += 1
            tokenized_examples["start_positions"].append(token_start_index - 1)
            while token_end_index >=0 and offsets[token_end_index][1] >= end_char:
                token_end_index -= 1
            tokenized_examples["end_positions"].append(token_end_index + 1)
            # assertion to check if this checks out
            context = dataset_dict['context'][sample_index]
            offset_st = offsets[tokenized_examples['start_positions'][-1]][0]
            offset_en = offsets[tokenized_examples['end_positions'][-1]][1]
            
            if context[offset_st : offset_en] != answer['text'][0]:
                inaccurate += 1

    total = len(tokenized_examples['id'])
    print(f"Preprocessing not completely accurate for {inaccurate}/{total} instances")
    return tokenized_examples



def read_and_process(args, tokenizer, dataset_dict, dir_name, dataset_name, split):
    #TODO: cache this if possible
    cache_path = f'{dir_name}/{dataset_name}_encodings.pt'
    if os.path.exists(cache_path) and not args.recompute_features:
        tokenized_examples = util.load_pickle(cache_path)
    else:
        if split=='train':
            tokenized_examples = prepare_train_data(dataset_dict, tokenizer)
        else:
            tokenized_examples = prepare_eval_data(dataset_dict, tokenizer)
        util.save_pickle(tokenized_examples, cache_path)
    return tokenized_examples


def get_dataset(args, datasets, data_dir, tokenizer, split_name):
    datasets = datasets.split(',')
    dataset_dict = None
    dataset_name=''
    for dataset in datasets:
        dataset_name += f'_{dataset}'
        dataset_dict_curr = util.read_squad(f'{data_dir}/{dataset}')
        dataset_dict = util.merge(dataset_dict, dataset_dict_curr)
    data_encodings = read_and_process(args, tokenizer, dataset_dict, data_dir, dataset_name, split_name)
    return util.QADataset(data_encodings, train=(split_name=='train')), dataset_dict

class MetaLearningTrainer():
    def __init__(self, base_model: torch.nn.Module, train_dir, val_dir, tokenizer, args, log):
        # meta-learning parameters
        self.meta_epochs = args.meta_epochs
        self.num_tasks = 3
        self.k_gradient_steps = 3
        self.meta_lr = args.meta_lr
        self.global_idx = 0
        self.path = os.path.join(args.save_dir, 'checkpoint')

        # base model parameters
        self.args = args
        self.log = log
        self.base_models = [DistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased").to(args.device)]\
                          * self.num_tasks
        self.meta_model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased").to(args.device)
        self.train_datasets = []
        self.train_dataset_probabilities = []
        self.train_dicts = []
        self.val_dataloader = None
        self.val_dict = None
        self.add_datasets(train_dir, tokenizer, 'train')
        self.add_datasets(val_dir, tokenizer, 'val')

        self.data_loaders = [DataLoader(train_dataset, batch_size=self.args.batch_size, sampler=RandomSampler(train_dataset))
                        for train_dataset in self.train_datasets]
        self.data_loaders_iterators = [iter(data_loader) for data_loader in self.data_loaders]
        self.data_loader_cursors = [0] * len(self.train_datasets)

        self.tbx = SummaryWriter(self.args.save_dir)

    def add_datasets(self, data_dir, tokenizer, split_name):
        data_paths = [os.path.basename(path) for path in Path(data_dir).glob('*') if not str(path).endswith('.pt')]
        if split_name == 'train':
            for data_path in data_paths:
                train_dataset, train_dict = get_dataset(self.args, data_path, data_dir, tokenizer, split_name)
                self.train_datasets.append(train_dataset)
                self.train_dataset_probabilities.append(len(train_dict))
                self.train_dicts.append(train_dict)
            total_num_entries = sum(self.train_dataset_probabilities)
            self.train_dataset_probabilities = [prob / total_num_entries for prob in self.train_dataset_probabilities]
        else:
            val_dataset, val_dict = get_dataset(self.args, ','.join(data_paths), data_dir, tokenizer, split_name)
            self.val_dataloader = DataLoader(val_dataset, batch_size=self.args.batch_size, sampler=SequentialSampler(val_dataset))
            self.val_dict = val_dict

    def evaluate(self, model, data_loader, data_dict, return_preds=False, split='validation'):
        device = self.args.device
        model.eval()
        all_start_logits = []
        all_end_logits = []
        with torch.no_grad(), \
                tqdm(total=len(data_loader.dataset)) as progress_bar:
            for batch in data_loader:
                # Setup for forward
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                batch_size = len(input_ids)
                outputs = model(input_ids, attention_mask=attention_mask)
                # Forward
                start_logits, end_logits = outputs.start_logits, outputs.end_logits
                # TODO: compute loss

                all_start_logits.append(start_logits)
                all_end_logits.append(end_logits)
                progress_bar.update(batch_size)

        # Get F1 and EM scores
        start_logits = torch.cat(all_start_logits).cpu().numpy()
        end_logits = torch.cat(all_end_logits).cpu().numpy()
        preds = util.postprocess_qa_predictions(data_dict,
                                                 data_loader.dataset.encodings,
                                                 (start_logits, end_logits))
        if split == 'validation':
            results = util.eval_dicts(data_dict, preds)
            results_list = [('F1', results['F1']),
                            ('EM', results['EM'])]
        else:
            results_list = [('F1', -1.0),
                            ('EM', -1.0)]
        results = OrderedDict(results_list)

        if return_preds:
            return preds, results
        return results

    def eval_helper(self, model, selected_index):
        train_dataloader = self.data_loaders[selected_index] = \
            DataLoader(self.train_datasets[selected_index], batch_size=self.args.batch_size,
                       sampler=RandomSampler(self.train_datasets[selected_index]))
        train_dict = self.train_dicts[selected_index]

        self.log.info(f'1Evaluating at step {self.global_idx}...')
        preds, curr_score = self.evaluate(model, train_dataloader, train_dict, return_preds=True)
        results_str = ', '.join(f'{k}: {v:05.2f}' for k, v in curr_score.items())
        self.log.info('Visualizing in TensorBoard...')
        for k, v in curr_score.items():
            self.tbx.add_scalar(f'val/{k}', v, self.global_idx)
        self.log.info(f'Eval {results_str}')
        if self.args.visualize_predictions:
            util.visualize(self.tbx,
                           pred_dict=preds,
                           gold_dict=self.val_dict,
                           step=self.global_idx,
                           split='val',
                           num_visuals=self.args.num_visuals)

    def train(self, model, selected_index):
        device = self.args.device
        optim = AdamW(model.parameters(), lr=self.args.lr)
        best_scores = {'F1': -1.0, 'EM': -1.0}

        with torch.enable_grad(), tqdm(total=self.k_gradient_steps) as progress_bar:
            for i in range(self.k_gradient_steps):
                if self.data_loader_cursors[selected_index] + 1 >= len(self.data_loaders[selected_index].dataset):
                    self.data_loaders[selected_index] = \
                        DataLoader(self.train_datasets[selected_index], batch_size=self.args.batch_size,
                                   sampler=RandomSampler(self.train_datasets[selected_index]))
                    self.data_loaders_iterators[selected_index] = iter(self.data_loaders[selected_index])
                    self.data_loader_cursors[selected_index] = 0

                optim.zero_grad()
                model.train()

                batch = next(self.data_loaders_iterators[selected_index])
                self.data_loader_cursors[selected_index] += self.args.batch_size

                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                start_positions = batch['start_positions'].to(device)
                end_positions = batch['end_positions'].to(device)
                outputs = model(input_ids, attention_mask=attention_mask,
                                start_positions=start_positions,
                                end_positions=end_positions)
                loss = outputs[0]
                loss.backward()
                optim.step()
                progress_bar.update(i + 1)
                progress_bar.set_postfix(NLL=loss.item())
                self.tbx.add_scalar('train/NLL', loss.item(), self.global_idx)
                if self.global_idx % self.args.eval_every == 0:
                    self.log.info(f'Evaluating at step {self.global_idx}...')
                    preds, curr_score = self.evaluate(model, self.val_dataloader, self.val_dict, return_preds=True)
                    results_str = ', '.join(f'{k}: {v:05.2f}' for k, v in curr_score.items())
                    self.log.info('Visualizing in TensorBoard...')
                    for k, v in curr_score.items():
                        self.tbx.add_scalar(f'val/{k}', v, self.global_idx)
                    self.log.info(f'Eval {results_str}')
                    if self.args.visualize_predictions:
                        util.visualize(self.tbx,
                                       pred_dict=preds,
                                       gold_dict=self.val_dict,
                                       step=self.global_idx,
                                       split='val',
                                       num_visuals=self.args.num_visuals)
                    if curr_score['F1'] >= best_scores['F1']:
                        best_scores = curr_score
                        self.meta_model.save_pretrained(self.path)
                self.global_idx += 1

    def update_meta_params(self):
        # meta_params = (1 - beta) * meta_params + beta * params_delta
        for meta_param in self.meta_model.parameters():
            meta_param.data.copy_(meta_param.data * (1 - self.meta_lr))
        for base_model in self.base_models:
            for meta_param, base_param in zip(self.meta_model.parameters(), base_model.parameters()):
                meta_param.data.copy_(meta_param.data + base_param.data * self.meta_lr / self.num_tasks)
        # propagate meta_params to base_params
        for base_model in self.base_models:
            for meta_param, base_param in zip(self.meta_model.parameters(), base_model.parameters()):
                base_param.data.copy_(meta_param.data)


    def meta_train(self):
        for epoch_num in range(self.meta_epochs):
            self.log.info(f'Epoch: {epoch_num}')
            selected_task_indices = np.random.choice(range(len(self.data_loaders)), self.num_tasks,
                                              p=self.train_dataset_probabilities)
            for i, selected_index in enumerate(selected_task_indices):
                # Train model on the current task
                self.train(self.base_models[i], selected_index)
            # Update meta-learning parameters and reset base model parameters.
            self.update_meta_params()

def main():
    # define parser and arguments
    args = get_train_test_args()

    util.set_seed(args.seed)
    checkpoint_path = os.path.join(args.save_dir, 'checkpoint')
    model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased")
    #model = DistilBertForQuestionAnswering.from_pretrained(checkpoint_path)
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

    if args.do_train:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        args.save_dir = util.get_save_dir(args.save_dir, args.run_name)
        log = util.get_logger(args.save_dir, 'log_train')
        log.info(f'Args: {json.dumps(vars(args), indent=4, sort_keys=True)}')
        log.info("Preparing Training Data...")
        args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        # args.device = torch.device('cpu')
        trainer = MetaLearningTrainer(
            model, train_dir=args.train_dir, val_dir=args.val_dir,
            tokenizer=tokenizer, args=args, log=log
        )
        trainer.meta_train()
    if args.do_eval:
        args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        split_name = 'test' if 'test' in args.eval_dir else 'validation'
        log = util.get_logger(args.save_dir, f'log_{split_name}')
        model = DistilBertForQuestionAnswering.from_pretrained(checkpoint_path)
        model.to(args.device)
        trainer = MetaLearningTrainer(
            model, train_dir="datasets/oodomain_train/", val_dir="datasets/oodomain_val",
            tokenizer=tokenizer, args=args, log=log
        )
        eval_dataset, eval_dict = get_dataset(args, args.eval_datasets, args.eval_dir, tokenizer, split_name)
        eval_loader = DataLoader(eval_dataset,
                                 batch_size=args.batch_size,
                                 sampler=SequentialSampler(eval_dataset))
        eval_preds, eval_scores = trainer.evaluate(model, eval_loader,
                                                   eval_dict, return_preds=True,
                                                   split=split_name)
        results_str = ', '.join(f'{k}: {v:05.2f}' for k, v in eval_scores.items())
        log.info(f'Eval {results_str}')
        # Write submission file
        sub_path = os.path.join(args.save_dir, split_name + '_' + args.sub_file)
        log.info(f'Writing submission file to {sub_path}...')
        with open(sub_path, 'w', newline='', encoding='utf-8') as csv_fh:
            csv_writer = csv.writer(csv_fh, delimiter=',')
            csv_writer.writerow(['Id', 'Predicted'])
            for uuid in sorted(eval_preds):
                csv_writer.writerow([uuid, eval_preds[uuid]])


if __name__ == '__main__':
    main()
