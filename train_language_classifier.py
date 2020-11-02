import datasets
from transformers import AutoTokenizer
import argparse
import torch.nn as nn
import torch
import timm
import torch.optim as optim
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import os
import data_utils
import json
import numpy as np
import language_models

train_hps = {
    'num_epochs' : 25,
    'lr' : 0.0001,
    'batch_size' : 32,
    'validate_every' : 500, # validates on small subset of val set
    'evaluate_every' : 5000, # evaluates on full test set using best ckpt
    'embedding_size' : 256,
    'hidden_size' : 256
}


def evaluate(dataloader, model, iter_no, max_batches = None):
    total_correct = 0.0
    total_examples = 0.0
    model.eval()
    for bidx, batch in enumerate(dataloader):
        sentence = batch['input_ids'].cuda()
        attention_mask = batch['attention_mask'].cuda()
        max_sentence_length = torch.max(torch.sum(attention_mask, dim=1)).item()
        labels = batch['label'].cuda()
        logits = model(sentence, max_sentence_length = max_sentence_length)
        prediction = torch.argmax(logits, dim = 1)
        correct = torch.sum(prediction == labels)
        total_correct += correct.item()
        total_examples += int(sentence.size(0))
        print("Evaluating", "{} out of {}".format(bidx, len(dataloader)))
        if (max_batches is not None) and bidx > max_batches:
            break
    acc = total_correct/total_examples
    model.train()
    
    return {
        'acc' : acc
    }

def save_checkpoint(model, learning_rate, acc, iteration, filepath):
    print("Saving model state at iteration {} to {}".format(iteration, filepath))
    torch.save({'iteration': iteration,
                'state_dict': model.state_dict(),
                'acc' : acc,
                'learning_rate': learning_rate}, filepath)

# def find_max_sentence_length(train_dataset, val_dataset):
#     max_length = 0
#     for row in train_dataset:
#         row_length = torch.sum(row['attention_mask']).item()
#         if row_length > max_length:
#             max_length = row_length
#     for row in val_dataset:
#         row_length = torch.sum(row['attention_mask']).item()
#         if row_length > max_length:
#             max_length = row_length
#     return max_length


def load_checkpoint(checkpoint_path, model):
    assert os.path.isfile(checkpoint_path)
    print("Loading checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint_dict['state_dict'])
    iteration = checkpoint_dict['iteration']
    print("Loaded checkpoint '{}' from iteration {}" .format(
        checkpoint_path, iteration))
    return model, iteration

def main():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--text_dataset', type=str)
    p.add_argument('--language_model', type=str)
    p.add_argument('--logdir', type=str, default = "/data2/paarth/ReprogrammingTransformers/ClassificationModels")
    
    args = p.parse_args()

    dataset_sentence_key_mapping = data_utils.dataset_sentence_key_mapping

    assert args.text_dataset in dataset_sentence_key_mapping

    train_dataset_raw = datasets.load_dataset(args.text_dataset, split="train")
    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
    text_key = dataset_sentence_key_mapping[args.text_dataset]
    train_dataset = train_dataset_raw.map(lambda e: tokenizer(e[text_key], truncation=True, padding='max_length'), batched=True)
    train_dataset = train_dataset.map(lambda e: data_utils.label_mapper(e, args.text_dataset), batched=True)
    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

    val_dataset_raw = datasets.load_dataset(args.text_dataset, split="test")
    val_dataset = val_dataset_raw.map(lambda e: tokenizer(e[text_key], truncation=True, padding='max_length'), batched=True)
    val_dataset = val_dataset.map(lambda e: data_utils.label_mapper(e, args.text_dataset), batched=True)
    val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

    # max_sentence_length = find_max_sentence_length(train_dataset, val_dataset)
    # print("Max sentence length", max_sentence_length)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_hps['batch_size'], shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=train_hps['batch_size'], shuffle=True)

    

    vocab_size = len(tokenizer.get_vocab())

    model = language_models.get_model(
        args.language_model, 
        vocab_size, 
        train_hps['embedding_size'], 
        train_hps['hidden_size'], 
        data_utils.dataset_num_classes[args.text_dataset]
    )

    
    model.cuda()
    
    optimizer = optim.Adam(model.parameters(), lr=train_hps['lr'])
    loss_criterion = nn.CrossEntropyLoss()

    exp_name = "classifier_{}_lr_{}_model_{}".format(
        args.text_dataset, train_hps['lr'], args.language_model
    )

    logdir = os.path.join(args.logdir, exp_name)
    ckptdir = os.path.join(logdir, "CKPTS")
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    if not os.path.exists(ckptdir):
        os.makedirs(ckptdir)

    tb_writer = SummaryWriter(logdir = logdir)

    iter_no = 0
    best_acc = 0.0
    best_iter_no = 0
    best_model_path = None
    prev_best_eval_iter = None
    for epoch in range(train_hps['num_epochs']):
        for bidx, batch in enumerate(train_loader):
            sentence = batch['input_ids'].cuda()
            attention_mask = batch['attention_mask'].cuda()
            max_sentence_length = torch.max(torch.sum(attention_mask, dim=1)).item()
            labels = batch['label'].cuda()
            logits = model(sentence, max_sentence_length = max_sentence_length)
            loss = loss_criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if iter_no % 10 == 0:
                print (iter_no, "Loss:", loss.item())
                tb_writer.add_scalar('train_loss', loss, iter_no)

            if iter_no % train_hps['validate_every'] == 0:
                print("Evaluating")
                metrics = evaluate(val_loader, 
                    model,
                    iter_no, max_batches = 100)
                tb_writer.add_scalar('val_acc', metrics['acc'], iter_no)
                print(metrics)
                model_path = os.path.join(ckptdir, "model.p")
                save_checkpoint(model, train_hps['lr'], metrics['acc'], iter_no, model_path)
                if metrics['acc'] >= best_acc:
                    best_model_path = os.path.join(ckptdir, "model_best.p")
                    save_checkpoint(model, train_hps['lr'], metrics['acc'], iter_no, best_model_path)
                    best_acc = metrics['acc']
                    best_iter_no = iter_no
                print("Best acc. till now:", best_acc, best_iter_no)

            if (iter_no + 1) % train_hps['evaluate_every'] == 0 and prev_best_eval_iter != best_iter_no:
                # Run evaluation on whole test set using the new best checkpoint (if found)
                print("Running full evaluation!")
                backup_ckpt_path = os.path.join(ckptdir, "model_temp.p")
                save_checkpoint(model, train_hps['lr'], metrics['acc'], iter_no, backup_ckpt_path)

                model, _ = load_checkpoint(best_model_path, model)
                metrics = evaluate(val_loader, 
                    model, 
                    iter_no)
                log_fn = os.path.join(logdir, "best_metrics.json")
                metrics['iter_no'] = best_iter_no
                with open(log_fn, "w") as f:
                    f.write(json.dumps(metrics))
                prev_best_eval_iter = best_iter_no
                model, _ = load_checkpoint(backup_ckpt_path, model)
                print("Ran full evaluation!")

            iter_no += 1

if __name__ == '__main__':
    main()
