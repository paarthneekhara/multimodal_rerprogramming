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