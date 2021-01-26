# Multi-modal Adversarial Reprogramming

## Installation
* Clone the repo: ``git clone multimodal_rerprogramming``
* Get the code for image models:
    1) ``git submodule init``
    2) ``git submodule update``
* Install TIMM (PyTorch image models):
    1) ``cd pytorch-image-models``
    2) ``pip install requirements.txt``
    3) ``pip install -e .``  (this installs library in editable mode)
* Install transformers and datasets library from huggingface and tensorboardX:
    1) ``pip install datasets``
    2) ``pip install transformers``
    3) ``pip install tensorboardX``
    3) ``pip install sklearn==0.23.2``
    

Sample Reprogrammer Checkpoint: https://drive.google.com/file/d/1xn3zm0DmNNVPEHb_fFLAWRoMjNpb-nIx/view?usp=sharing

Classification Datasets: https://archive.ics.uci.edu/ml/datasets.php?format=&task=cla&att=&area=&numAtt=&numIns=&type=seq&sort=attTypeUp&view=table

## Reprogramming Experiment Commands

CUDA_VISIBLE_DEVICES=0 python reprogramming.py --text_dataset TEXTDATSET --logdir <PATH WHERE CKPTS/TB LOG WILL BE SAVED>  --cache_dir <PATH WHERE HF CACHE WILL BE CREATED> --reg_alpha 1e-4 --pretrained_vm 1 --resume_training 1 --use_char_tokenizer 0 --img_patch_size 16; 
    
* Pretrained VM 1 or 0 depending on pretrained or random network
* Set use_char_tokenizer to 1 if you using DNA datasets - protein_splice, geneh3
* exp_name_extension name of the experiment that uniquely identifies your run
* Set -resume_training to 1 if you want to continue a run that stopped. 
* img_patch_size Lookup the google sheet for this
* For bounded experiments: Supply --base_image_path library.jpg --max_iterations 150000

