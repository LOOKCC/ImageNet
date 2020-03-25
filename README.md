# ImageNet 
This is a ImageNet Example for classification task.

## Installation
I suggest you use conda to create a environment and use pip to install packges.(Because pip is faster than conda especially use this mirror https://pypi.tuna.tsinghua.edu.cn/simple/)
```
git clone https://github.com/LOOKCC/ImageNet.git
pip install torch torchvision -i https://pypi.tuna.tsinghua.edu.cn/simple/
pip install tensorboard
pip install PyYAML
```

## Config
config is at config.yaml, They are commonly used configurations, such as lr or batch_szie.
There are some important config need to be explanded.
 - `fix_fc` Because imagenet has 1000 classes, when your task has 2 classes, you need to fix last fc layer. But if the fc is 2, you can't load imagenet pretrain models. So you need to fix fc layer after load imagenet checkpoint.
 - `max_epoch`  and  `max_iteration` Your training will stop when epoch reach max_epoch or iteration reach max_iteration. So you should make both of them a lager number to avoid stopping early.
 - lr just follow this formula, and `lr_lowest` represents the lowest lr.  
 ```
 lr = args.base_lr * (args.lr_decay_factor ** (epoch // args.lr_decay_epoch))
 ```
 - `validation_freq` controls how many iterations to test.


## Dataset
In config, there are two text file represented train_set and test_set.
In text file, One line represents one sample. They format as follows:
```
path_to_image/example.txt label
``` 
The tpye of label is int, for example, if it's a Binary classification problem, then label is 0 or 1.

## Usage
```
usage: main.py [-h] [-c CONFIG] [--resume PATH] [-e] [--save SAVE]

PyTorch ImageNet Training

optional arguments:
  -h, --help            show this help message and exit
  -c CONFIG, --config CONFIG
                        path to config file.
  --resume PATH         path to latest checkpoint (default: none)
  -e, --evaluate        evaluate model on validation set
  --save SAVE           save_dir
```
tips: you may need to use `CUDA_VISIBLE_DEVICES`. Because the default device is `cuda:0`  
sample example:
```
CUDA_VISIBLE_DEVICE=2,3 python main.py -c config.yaml --save save_test
```

## Tensorboard
If you want to use Tensorboard:
```
tesorboard --logdir=. --port=8090 --blind_all
```

## Add Models
If you want to add your Net, such as Resnet. You need to do followings:  
1. Add your_net.py in ./net
2. In your_net.py, us `__all__`  to set interface. if you don't konw about this, just do this link do: https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
3. modify `./net/__init__.py`, add `from xxx import xxx`
4. modify confi: arch to the net name

## Result
If your --save is save_test, there will be a file called report.txt be created. 
```
iter train_loss train_acc test_loss test_acc lr
1000 0.5406224970574622 87.31268310546875 0.4301627216339111 86.00000762939453 0.1
2000 0.48305593270888514 87.4562759399414 0.41288787841796876 86.00000762939453 0.1
```
This informaion will help you to choose checkpoints.

## Finally
I will very appreciate if this repository helps your training. GL&HF.
