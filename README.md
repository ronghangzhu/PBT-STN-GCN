# PBT-STN-GCN
This is the code for paper ["Automated Graph Learning via Population based Self-Tuning GCN"](https://dl.acm.org/doi/abs/10.1145/3404835.3463056?casa_token=XXMSM5ooVC4AAAAA:17WWDcs8gNlboXpM5t9XVyCcAUoKICtIpF-fxTFcr-WOYMI7XUPXKGED6SoqS3dbMg2LY8XNW4Nm). It only includes STN-GCN code, we will release the PBT-STN-GCN code later.


# Prerequisites
* Python3
* Pytorch == 1.0.0 (with suitable CUDA and CuDNN version)

# Dataset
The datasets (Cora, Citeseer and PubMed) are in [GoogleDrive](https://drive.google.com/file/d/1TXVTe2saZ80d26X5zhkqObhfhhTm6vyl/view?usp=sharing) and [BaiduPan (pw:frvg)](https://pan.baidu.com/s/1d5D5qApPvlYVdV5qWlUIgA).  
You need to create a "./data" file in "./core" file and move the dataset into it.

# Training
You can run `python train_new.py` to train and evaluate.

# Citation
If you use this code for you research, please consider citing:  
```
@inproceedings{zhu2021automated,
  title={Automated Graph Learning via Population Based Self-Tuning GCN},
  author={Zhu, Ronghang and Tao, Zhiqiang and Li, Yaliang and Li, Sheng},
  booktitle={Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval},
  pages={2096--2100},
  year={2021}
}
```
