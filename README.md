# Cross-modal Adversarial Reprogramming

Code for our WACV 2022 paper [Cross-modal Adversarial Reprogramming](https://arxiv.org/abs/2102.07325). 

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
    

<!-- Sample Reprogrammer Checkpoint: https://drive.google.com/file/d/1xn3zm0DmNNVPEHb_fFLAWRoMjNpb-nIx/view?usp=sharing

Classification Datasets: https://archive.ics.uci.edu/ml/datasets.php?format=&task=cla&att=&area=&numAtt=&numIns=&type=seq&sort=attTypeUp&view=table -->

## Running the Experiments

The text/sequence dataset configurations are defined in ``data_utils.py``. We can either use text-classification [datasets available in the huggingface hub](https://huggingface.co/docs/datasets/) or use our custom datasets (defined as json files) with the same API. To reprogram an image model for a text classification task run:

CUDA_VISIBLE_DEVICES=0 python reprogramming.py --text_dataset TEXTDATSET --logdir <PATH WHERE CKPTS/TB LOG WILL BE SAVED>  --cache_dir <PATH WHERE HF CACHE WILL BE CREATED> --reg_alpha 1e-4 --pretrained_vm 1 --resume_training 1 --use_char_tokenizer 0 --img_patch_size 16 --vision_model tf_efficientnet_b4; 

* TEXTDATSET is one of the dataset keys defined in data_utils.py
* Pretrained VM 1 or 0 depending on pretrained or random network
* Set use_char_tokenizer to 1 if you using DNA datasets - protein_splice, geneh3
* exp_name_extension name of the experiment that uniquely identifies your run
* Set --resume_training to 1 if you want to continue a run that stopped. 
* img_patch_size: Image patch size to embed each sequence token into
* For bounded experiments: Supply --base_image_path library.jpg --max_iterations 150000 (or any other image)
* vision_model is one of the following [vit_base_patch16_384, tf_efficientnet_b4, resnet50, tf_efficientnet_b7, inception_v3] : some configurations for these models are defined in data_utils.py
    
Once the model is trained, you may use the ``InferenceNotebook.ipynb`` notebook to visualize the reprogrammed images etc. Accuracy and other metrics on the test set are logged in tensorboard during training. 

## Citing our work

```
@inproceedings{neekhara2022crossmodal,
  title={Cross-modal Adversarial Reprogramming},
  author={Neekhara, Paarth and Hussain, Shehzeen and Du, Jinglong and Dubnov, Shlomo and Koushanfar, Farinaz and McAuley, Julian },
  booktitle={WACV},
  year={2022}
}
```
