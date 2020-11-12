import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from torchvision.transforms import ToTensor, Resize, Normalize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
        sentence_embedding = torch.tanh(self.token_embedding(sentence_batch)) # (N, l, 16*16*3)
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