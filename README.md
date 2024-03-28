当前代码只能在node01上运行

# Environment

```
git clone https://github.com/ceastld/adgs.git --recursive
cd adgs

conda create --name adgs -y python=3.10
conda activate adgs

pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
# slow
pip install -r requirements.txt
```

此环境配置参考了 https://github.com/ShenhanQian/GaussianAvatars

# Data

目前直接使用 xj 学长处理好的dataset

```
ln -s /xiangjun/gaussian-head/dataset/ dataset
```

# train

````
bash train.sh
```

log目录为 dataset/id2_25/log_ldy


some reference codes 

https://github.com/EvelynFan/FaceFormer/blob/main/wav2vec.py


# FlashAvatar

[Paper](https://arxiv.org/abs/2312.02214)|[Project Page](https://ustc3dv.github.io/FlashAvatar/)

![teaser](exhibition/teaser.png)
Given a monocular video sequence, our proposed FlashAvatar can reconstruct a high-fidelity digital avatar in minutes which can be animated and rendered over 300FPS at the resolution of 512×512 with an Nvidia RTX 3090.


## TODOLIST

* 数据预处理（包括flame，语音）
* 语音到表情系数，直接用MLP吧，不管合不合适

* 解决tensorboard无法使用的问题