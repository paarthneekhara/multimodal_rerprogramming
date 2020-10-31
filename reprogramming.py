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


train_hps = {
    'num_epochs' : 50,
    'lr' : 2e-3,
    'batch_size' : 4,
    'evaluate_every' : 500
}

class ReprogrammingFuntion(nn.Module):
    def __init__(self, vocab_size, img_patch_size = 16, img_size = 384):
        super(ReprogrammingFuntion, self).__init__()

        assert img_size % img_patch_size == 0
        self.img_patch_size = img_patch_size
        self.img_size = img_size
        self.token_embedding = nn.Embedding(vocab_size, img_patch_size * img_patch_size * 3)
        self.num_patches_row = int(img_size/img_patch_size)
        self.num_patches = self.num_patches_row * self.num_patches_row


    def forward(self, sentence_batch):
        sentence_embedding = F.tanh(self.token_embedding(sentence_batch)) # (N, l, 16*16*3)
        _N, _L, _ = sentence_embedding.size()
        sentence_embedding = sentence_embedding.view(_N, _L, 3, self.img_patch_size, self.img_patch_size)

        reprogrammed_image = torch.zeros(_N, 3, self.img_size, self.img_size).cuda()
        for patch_idx in range(self.num_patches):
            i_start = int(patch_idx / self.num_patches_row) * 16
            j_start = (patch_idx % self.num_patches_row) * 16
            i_end = i_start + 16
            j_end = j_start + 16
            if patch_idx < _L:
                reprogrammed_image[:,:,i_start:i_end,j_start:j_end] = sentence_embedding[:,patch_idx]
            else:
                # adding the padding embedding all the way till the end
                reprogrammed_image[:,:,i_start:i_end,j_start:j_end] = sentence_embedding[:,_L-1]

        
        return reprogrammed_image


def evaluate(dataloader, vision_model, reprogrammer, tb_writer, iter_no, max_batches = None):
    total_correct = 0.0
    total_examples = 0.0
    reprogrammer.eval()
    for bidx, batch in enumerate(dataloader):
        sentence = batch['input_ids'].cuda()
        labels = batch['label'].cuda()
        programmed_img = reprogrammer(sentence)
        if bidx == 0:
            tb_writer.add_image("ProgrammedImage", programmed_img[0], iter_no)
        logits = vision_model(programmed_img)
        prediction = torch.argmax(logits, dim = 1)
        correct = torch.sum(prediction == labels)
        total_correct += correct.item()
        total_examples += int(sentence.size(0))
        print("Evaluating", "{} out of {}".format(bidx, len(dataloader)))
        if (max_batches is not None) and bidx > max_batches:
            break
    acc = total_correct/total_examples
    reprogrammer.train()
    return {
        'acc' : acc
    }


def save_checkpoint(model, learning_rate, acc, iteration, filepath):
    print("Saving model state at iteration {} to {}".format(iteration, filepath))
    torch.save({'iteration': iteration,
                'state_dict': model.state_dict(),
                'acc' : acc,
                'learning_rate': learning_rate}, filepath)

def main():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--text_dataset', type=str)
    p.add_argument('--logdir', type=str, default = "/data2/paarth/ReprogrammingTransformers")
    p.add_argument('--img_patch_size', type=int, default = 16)
    p.add_argument('--img_size', type=int, default = 384)
    p.add_argument('--vision_model', type=str, default = 'vit_base_patch16_384')
    args = p.parse_args()

    dataset_sentence_key_mapping = data_utils.dataset_sentence_key_mapping

    assert args.text_dataset in dataset_sentence_key_mapping

    train_dataset_raw = datasets.load_dataset(args.text_dataset, split="train")
    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
    text_key = dataset_sentence_key_mapping[args.text_dataset]
    train_dataset = train_dataset_raw.map(lambda e: tokenizer(e[text_key], truncation=True, padding='max_length'), batched=True)
    train_dataset = train_dataset.map(lambda e: data_utils.label_mapper(e, args.text_dataset), batched=True)
    train_dataset.set_format(type='torch', columns=['input_ids', 'label'])

    val_dataset_raw = datasets.load_dataset(args.text_dataset, split="test")
    val_dataset = val_dataset_raw.map(lambda e: tokenizer(e[text_key], truncation=True, padding='max_length'), batched=True)
    val_dataset = val_dataset.map(lambda e: data_utils.label_mapper(e, args.text_dataset), batched=True)
    val_dataset.set_format(type='torch', columns=['input_ids', 'label'])


    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_hps['batch_size'], shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=train_hps['batch_size'], shuffle=True)

    vision_model = timm.create_model(args.vision_model, pretrained=True)
    vision_model.eval()
    vision_model.cuda()

    vocab_size = len(tokenizer.get_vocab())
    reprogrammer = ReprogrammingFuntion(vocab_size, args.img_patch_size, args.img_size)
    reprogrammer.cuda()
    
    optimizer = optim.Adam(reprogrammer.parameters(), lr=train_hps['lr'])
    loss_criterion = nn.CrossEntropyLoss()

    exp_name = "dataset_{}_lr_{}".format(args.text_dataset, train_hps['lr'])
    logdir = os.path.join(args.logdir, exp_name)
    ckptdir = os.path.join(logdir, "CKPTS")
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    if not os.path.exists(ckptdir):
        os.makedirs(ckptdir)

    tb_writer = SummaryWriter(logdir = logdir)

    iter_no = 0
    best_acc = 0.0
    for epoch in range(train_hps['num_epochs']):
        for bidx, batch in enumerate(train_loader):
            sentence = batch['input_ids'].cuda()
            labels = batch['label'].cuda()
            programmed_img = reprogrammer(sentence)
            logits = vision_model(programmed_img)
            loss = loss_criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if iter_no % 10 == 0:
                print (iter_no, "Loss:", loss.item())
                tb_writer.add_scalar('train_loss', loss, iter_no)

            if iter_no % train_hps['evaluate_every'] == 0:
                print("Evaluating")
                metrics = evaluate(val_loader, vision_model, 
                    reprogrammer, tb_writer, 
                    iter_no, max_batches = 100)
                tb_writer.add_scalar('val_acc', metrics['acc'], iter_no)
                print(metrics)
                model_path = os.path.join(ckptdir, "model.p")
                save_checkpoint(reprogrammer, train_hps['lr'], metrics['acc'], iter_no, model_path)
                if metrics['acc'] > best_acc:
                    model_path = os.path.join(ckptdir, "model_best.p")
                    save_checkpoint(reprogrammer, train_hps['lr'], metrics['acc'], iter_no, model_path)
                    best_acc = metrics['acc']
                

            iter_no += 1




if __name__ == '__main__':
    main()
