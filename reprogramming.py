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
from torchvision import transforms
from PIL import Image
from torchvision.transforms import ToTensor, Resize, Normalize
import json
import numpy as np
import pprint

train_hps = {
    'num_epochs' : 100,
    'max_iterations' : 300000,
    'lr' : 0.001, # overridden by args
    'batch_size' : 4,
    'validate_every' : 500, # validates on small subset of val set
    'evaluate_every' : 5000, # evaluates on full test set using best ckpt
    'label_reduction' : 'max' # overridden by args
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def unnormalize_image(tensor, mean, std):
    """
    tensor: Normalized image of shape (nc, h, w)
    """
    mean = torch.tensor(mean)[:,None,None].to(device)
    std = torch.tensor(std)[:,None,None].to(device)
    return tensor * std + mean

def normalize_image(tensor, mean, std):
    """
    tensor: Unnormalized image of shape (nc, h, w)
    """
    mean = torch.tensor(mean)[:,None,None].to(device)
    std = torch.tensor(std)[:,None,None].to(device)
    return (tensor - mean) / std

def get_mapped_logits(logits, class_mapping):
    """
    logits : Tensor of shape (batch_size, 1000) # imagenet class logits
    class_mapping: class_mapping[i] = list of image net labels for text class i
    reduction : max or mean
    """
    reduction = train_hps['label_reduction']
    mapped_logits = []
    for class_no in range(len(class_mapping)):
        if reduction == "max":
            class_logits, _ = torch.max(logits[:,class_mapping[class_no]], dim = 1) # batch size
        elif reduction == "mean":
            class_logits = torch.mean(logits[:,class_mapping[class_no]], dim = 1) # batch size
        else:
            raise NotImplentedException()

        mapped_logits.append(class_logits)
    return torch.stack(mapped_logits, dim = 1)

def create_label_mapping(n_classes, m_per_class, image_net_labels = None):
    """
    n_classes: No. of classes in text dataset
    m_per_class: Number of imagenet labels to be mapped to each text class
    """
    if image_net_labels is None:
        image_net_labels = range(1000)

    class_mapping = [[] for i in range(n_classes)]

    idx = 0
    for _m in range(m_per_class):
        for _class_no in range(n_classes):
            class_mapping[_class_no].append(image_net_labels[idx])
            idx += 1
    return class_mapping

def get_imagenet_label_list(vision_model, base_image, img_size):
    if base_image is None:
        torch.manual_seed(42)
        base_image = 2 * torch.rand(3, img_size, img_size).to(device) - 1.0

    logits = vision_model(base_image[None])[0]
    label_sort = torch.argsort(logits)
    label_list = label_sort.detach().cpu().numpy().tolist()

    return label_list


class ReprogrammingFuntion(nn.Module):
    def __init__(self, vocab_size, img_patch_size = 16, img_size = 384, 
        img_path=None, alpha=0.2, 
        img_mean=(0.5, 0.5, 0.5), img_std=(0.5, 0.5, 0.5)):
        super(ReprogrammingFuntion, self).__init__()

        assert img_size % img_patch_size == 0
        self.img_patch_size = img_patch_size
        self.img_size = img_size
        self.token_embedding = nn.Embedding(vocab_size, img_patch_size * img_patch_size * 3)
        self.num_patches_row = int(img_size/img_patch_size)
        self.num_patches = self.num_patches_row * self.num_patches_row
        self.base_image = None
        if img_path is not None:
            image = Image.open(img_path)
            transform=transforms.Compose([
                                  Resize((img_size, img_size)),
                                  ToTensor(),
                                Normalize(img_mean,img_std),
                                ])

            image = transform(image) 
            self.base_image = torch.tensor(image, requires_grad=False).to(device)
            self.alpha = alpha


    def forward(self, sentence_batch):
        sentence_embedding = F.tanh(self.token_embedding(sentence_batch)) # (N, l, 16*16*3)
        _N, _L, _ = sentence_embedding.size()
        sentence_embedding = sentence_embedding.view(_N, _L, 3, self.img_patch_size, self.img_patch_size)

        reprogrammed_image = torch.zeros(_N, 3, self.img_size, self.img_size).to(device)
        
        for patch_idx in range(self.num_patches):
            i_start = int(patch_idx / self.num_patches_row) * self.img_patch_size
            j_start = (patch_idx % self.num_patches_row) * self.img_patch_size
            i_end = i_start + self.img_patch_size
            j_end = j_start + self.img_patch_size
            if patch_idx < _L:
                reprogrammed_image[:,:,i_start:i_end,j_start:j_end] = sentence_embedding[:,patch_idx]
            else:
                # adding the padding embedding all the way till the end
                reprogrammed_image[:,:,i_start:i_end,j_start:j_end] = sentence_embedding[:,_L-1]

        if self.base_image is not None:
            base_image_batch = self.base_image[None].repeat((_N, 1, 1, 1))
            reprogrammed_image = base_image_batch + self.alpha * reprogrammed_image
        
        reprogrammed_image = torch.clamp(reprogrammed_image, -1.0, 1.0) # because image is normalized
        
        return reprogrammed_image


def evaluate(dataloader, vision_model, reprogrammer, tb_writer, 
    iter_no, class_mapping, max_batches = None, 
    img_mean=(0.5, 0.5, 0.5), img_std=(0.5, 0.5, 0.5)):
    
    total_correct = 0.0
    total_examples = 0.0
    reprogrammer.eval()
    l_inf_norms = []
    for bidx, batch in enumerate(dataloader):
        sentence = batch['input_ids'].to(device)
        labels = batch['label'].to(device)
        programmed_img = reprogrammer(sentence)
        if bidx == 0:
            vis_image = unnormalize_image(programmed_img[0], img_mean, img_std)
            tb_writer.add_image("ProgrammedImage", vis_image, iter_no)
        if reprogrammer.base_image is not None:
            _N = sentence.size(0)
            base_image_batch = reprogrammer.base_image[None].repeat((_N, 1, 1, 1))
            pert_normalized = programmed_img - base_image_batch
            pert_unnormalized = unnormalize_image(pert_normalized, img_mean, img_std)
            _l_inf_norm = torch.max(torch.abs(pert_unnormalized)).item()
            l_inf_norms.append(_l_inf_norm)

        logits = vision_model(programmed_img)
        mapped_logits = get_mapped_logits(logits, class_mapping)
        prediction = torch.argmax(mapped_logits, dim = 1)
        correct = torch.sum(prediction == labels)
        total_correct += correct.item()
        total_examples += int(sentence.size(0))
        print("Evaluating", "{} out of {}".format(bidx, len(dataloader)))
        if (max_batches is not None) and bidx > max_batches:
            break
    acc = total_correct/total_examples
    mean_linf_norm = None
    if len(l_inf_norms) > 0:
        mean_linf_norm = float(np.mean(l_inf_norms))

    reprogrammer.train()
    
    return {
        'acc' : acc,
        'mean_linf_norm' : mean_linf_norm
    }


def save_checkpoint(model, learning_rate, acc, iteration, image_net_labels, class_mapping, filepath):
    print("Saving model state at iteration {} to {}".format(iteration, filepath))
    torch.save({'iteration': iteration,
                'state_dict': model.state_dict(),
                'acc' : acc,
                'image_net_labels' : image_net_labels,
                'class_mapping' : class_mapping,
                'learning_rate': learning_rate}, filepath)

def load_checkpoint(checkpoint_path, model):
    assert os.path.isfile(checkpoint_path)
    print("Loading checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint_dict['state_dict'])
    iteration = checkpoint_dict['iteration']

    image_net_labels = None
    if 'image_net_labels' in checkpoint_dict:
        image_net_labels = checkpoint_dict['image_net_labels']

    print("Loaded checkpoint '{}' from iteration {}" .format(
        checkpoint_path, iteration))
    return model, iteration, image_net_labels

def main():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--text_dataset', type=str)
    p.add_argument('--logdir', type=str, default = "/data2/paarth/ReprogrammingTransformers/ReprogrammingModels")
    p.add_argument('--cache_dir', type=str, default = "/data2/paarth/HuggingFaceDatasets")
    p.add_argument('--img_patch_size', type=int, default = 16)
    p.add_argument('--img_size', type=int, default = 384)
    p.add_argument('--vision_model', type=str, default = 'vit_base_patch16_384')
    p.add_argument('--base_image_path', type=str, default = None)
    p.add_argument('--label_reduction', type=str, default = 'max')
    p.add_argument('--pert_alpha', type=float, default = 0.2)
    p.add_argument('--lr', type=float, default = 0.001)
    p.add_argument('--resume_training', type=int, default = 0)
    p.add_argument('--m_per_class', type=int, default = 1)
    args = p.parse_args()

    train_hps['lr'] = args.lr
    train_hps['label_reduction'] = args.label_reduction

    dataset_sentence_key_mapping = data_utils.dataset_sentence_key_mapping

    assert args.text_dataset in dataset_sentence_key_mapping

    train_dataset_raw = datasets.load_dataset(args.text_dataset, split="train", cache_dir = args.cache_dir)
    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
    text_key = dataset_sentence_key_mapping[args.text_dataset]
    train_dataset = train_dataset_raw.map(lambda e: tokenizer(e[text_key], truncation=True, padding='max_length'), batched=True)
    train_dataset = train_dataset.map(lambda e: data_utils.label_mapper(e, args.text_dataset), batched=True)
    train_dataset.set_format(type='torch', columns=['input_ids', 'label'])

    val_dataset_raw = datasets.load_dataset(args.text_dataset, split="test", cache_dir = args.cache_dir)
    val_dataset = val_dataset_raw.map(lambda e: tokenizer(e[text_key], truncation=True, padding='max_length'), batched=True)
    val_dataset = val_dataset.map(lambda e: data_utils.label_mapper(e, args.text_dataset), batched=True)
    val_dataset.set_format(type='torch', columns=['input_ids', 'label'])


    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_hps['batch_size'], shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=train_hps['batch_size'], shuffle=True)

    vision_model = timm.create_model(args.vision_model, pretrained=True)
    vision_model.eval()
    vision_model.to(device)

    vocab_size = len(tokenizer.get_vocab())

    img_mean = data_utils.image_model_configs[args.vision_model]['mean']
    img_std = data_utils.image_model_configs[args.vision_model]['std']

    n_classes = data_utils.dataset_num_classes[args.text_dataset]
    

    reprogrammer = ReprogrammingFuntion(vocab_size, args.img_patch_size, 
        args.img_size, 
        img_path = args.base_image_path, alpha = args.pert_alpha, img_mean = img_mean, img_std = img_std)
    reprogrammer.to(device)
    image_net_labels = get_imagenet_label_list(vision_model, reprogrammer.base_image, args.img_size)
    print("Imagenet Label Ordering..", image_net_labels[:20])
    optimizer = optim.Adam(reprogrammer.parameters(), lr=train_hps['lr'])
    loss_criterion = nn.CrossEntropyLoss()

    base_image_name = None
    if args.base_image_path is not None:
        base_image_name = args.base_image_path.split("/")[-1].split(".")[0]
    exp_name = "ds_{}_lr_{}_bimg_{}_vm_{}_alpha_{}_m_label_{}_{}".format(
        args.text_dataset, train_hps['lr'], base_image_name, 
        args.vision_model, args.pert_alpha, args.m_per_class, args.label_reduction
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

    if args.resume_training == 1:
        resume_model_path = os.path.join(ckptdir, "model.p")
        print("Resuming from ckpt", resume_model_path)
        if not os.path.exists(resume_model_path):
            raise Exception("model not found")
        reprogrammer, iter_no, image_net_labels = load_checkpoint(resume_model_path, reprogrammer)


    class_mapping = create_label_mapping(n_classes, args.m_per_class, image_net_labels)
    print("Class Mapping")
    pprint.pprint(class_mapping)

    for epoch in range(train_hps['num_epochs']):
        for bidx, batch in enumerate(train_loader):
            sentence = batch['input_ids'].to(device)
            labels = batch['label'].to(device)
            programmed_img = reprogrammer(sentence)
            logits = vision_model(programmed_img)
            mapped_logits = get_mapped_logits(logits, class_mapping)
            loss = loss_criterion(mapped_logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if iter_no % 10 == 0:
                print (iter_no, "Loss:", loss.item())
                tb_writer.add_scalar('train_loss', loss, iter_no)

            if iter_no % train_hps['validate_every'] == 0:
                print("Evaluating")
                metrics = evaluate(val_loader, vision_model, 
                    reprogrammer, tb_writer, 
                    iter_no, class_mapping, max_batches = 100, img_mean = img_mean, img_std = img_std)
                tb_writer.add_scalar('val_acc', metrics['acc'], iter_no)
                print(metrics)
                model_path = os.path.join(ckptdir, "model.p")
                save_checkpoint(reprogrammer, train_hps['lr'], metrics['acc'], iter_no, image_net_labels, class_mapping, model_path)
                if metrics['acc'] >= best_acc:
                    best_model_path = os.path.join(ckptdir, "model_best.p")
                    save_checkpoint(reprogrammer, train_hps['lr'], metrics['acc'], iter_no, image_net_labels, class_mapping, best_model_path)
                    best_acc = metrics['acc']
                    best_iter_no = iter_no
                print("Best acc. till now:", best_acc, best_iter_no)

            if (iter_no + 1) % train_hps['evaluate_every'] == 0 and prev_best_eval_iter != best_iter_no:
                # Run evaluation on whole test set using the new best checkpoint (if found)
                print("Running full evaluation!")
                backup_ckpt_path = os.path.join(ckptdir, "model_temp.p")
                save_checkpoint(reprogrammer, train_hps['lr'], metrics['acc'], iter_no, image_net_labels, class_mapping, backup_ckpt_path)

                reprogrammer, _, _ = load_checkpoint(best_model_path, reprogrammer)
                metrics = evaluate(val_loader, vision_model, 
                    reprogrammer, tb_writer, 
                    iter_no, class_mapping, img_mean = img_mean, img_std = img_std)
                log_fn = os.path.join(logdir, "best_metrics.json")
                metrics['iter_no'] = best_iter_no
                metrics['base_image_path'] = args.base_image_path
                metrics['alpha'] = args.pert_alpha
                metrics['train_hps'] = train_hps
                with open(log_fn, "w") as f:
                    f.write(json.dumps(metrics))
                prev_best_eval_iter = best_iter_no
                reprogrammer, _, _ = load_checkpoint(backup_ckpt_path, reprogrammer)
                print("Ran full evaluation!")

            iter_no += 1
            if iter_no > train_hps['max_iterations']:
                break

if __name__ == '__main__':
    main()
