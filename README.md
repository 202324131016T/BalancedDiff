# BalancedDiff

> **BalancedDiff: Balanced Diffusion Network for High-quality Molecule Generation**
>
> Author: Yulong Wu, Jin Xie, Jing Nie, Bonan Ding, Yuansong Zeng and Jiale Cao

## Environment

```
conda create -n BalancedDiff python=3.8
conda activate BalancedDiff
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install matplotlib  -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install tqdm  -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install pandas  -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install scikit-learn  -i https://pypi.tuna.tsinghua.edu.cn/simple

Other requirements:
- Linux
- NVIDIA GPU
- The detailed environment configuration is shown in environment.yml.
```

## Dataset

```
The dataset is stored in the ./data/ directory, and detailed information can be found in ./data/readme.txt.
```

## Run

```
Run the run.sh directly to perform Training, Sampling, and Evaluation. 

The inference.py provides additional methods for molecular property evaluation.
```

## Other information

```
The ./models/our_model_main.py is the main model of our method.

The ./result/checkpoint/best.pt is the checkpoint of our main method.

The ./result/filter_mols/ provides some result of our method.

More results can be found in ./result/readme.txt.
```



