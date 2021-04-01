# meibo-ML: Meibography phenotyping and classification from unsupervised discriminative feature learning

by Chun-Hsiao Yeh, Stella X. Yu and Meng C. Lin at School of Optometry and ICSI, UC Berkeley

<em>Translational Vision Science & Technology February 2021, Vol.10, 4.</em>

[Journal Page](https://tvst.arvojournals.org/article.aspx?articleid=2772251) | [PDF](http://www1.icsi.berkeley.edu/~stellayu/publication/doc/2021meiboTVST.pdf) | 

<img src="title-img.png" width="ˊ%" />

The proposed NPID approach automatically analyses MG atrophy severity from meibography images without prior image annotations, and categorizes the gland characteristics through hierarchical clustering. This method provides quantitative information on the MG atrophy severity based on the analysis of phenotypes. Please contact [Chun-Hsiao Yeh](mailto:daniel-yeh@berkeley.edu) for further details or information.


## Updates
[3/2021] Initial Commit. We re-implemented meibo-ML in this repo. (instruction below).


## Requirements
### Packages
* Python = 3.7
* PyTorch >= 1.6
* pandas
* numpy

## Prepare Datasets
It is recommended to follow your dataset root (assuming $YOUR_DATA_ROOT) to $meibo_ML/data. If your folder structure is different, you may need to change the corresponding paths in files. Note that the meibography data we used is not published yet, we will keep updated once the data is open.
```
meibo-ML
├── data
│   ├── train
│   ├── val
│   └── test
```

## Usage
This section provides basic tutorials about the usage of meibo-ML implementation.

### Pretraining
To pretrain the model on the meibography data with a single GPU, try the following command:
```
python main.py 'data/' \
--resume lemniscate_resnet50.pth \
--arch resnet50 \
-j 32 \
--nce-k 4096 \
--nce-t 0.07 \
--lr 0.005 \
--nce-m 0.5 \
--low-dim 128 \
--epochs 200 \
--wd 4e-5 \
--save_schedule 30 60 90 120 150 180 200 \
--save_path ${your experiment folder} \
-b 32
```
Note: `--resume` is set to resume ImageNet pretrained model. Our Initial setting is to train a ResNet-50 backbone network with the batchsize of 32, initial learning rate of 0.005, and the temperature of 0.07. `--save_path` specifies the output folder, and the checkpoints will be dumped to '${your experiment folder}'.  `--save_schedule` is to save extra checkpoints for further evaluation.

### Evaluation
To test the model on the meibography data, try the following command:
```
python main.py 'data/' \
--resume '${your saved model for evalution}' \
--arch resnet50 \
-e
```
Note: `--resume` is to specify the saved models for evaluation. ('${your saved model for evalution}')

### Pretrained Models
The pre-trained models can be found below. Note that for the best performance we used ImageNet pretrained model in our training.

| Model checkpoint and hub-module   |          Top-1         |
|-----------------------------------|------------------------|
|[Our meibo-ML w ImageNet pretrained](https://drive.google.com/file/d/1mWFL47DeHbKOX-VLVXPQ00nDRswaKdd0/view?usp=sharing) |          79.4          |


## Citation
please cite our work if you use this work in your research.
```
@article{yeh2021meibography,
  title={Meibography phenotyping and classification from unsupervised discriminative feature learning},
  author={Yeh, Chun-Hsiao and Stella, X Yu and Lin, Meng C},
  journal={Translational Vision Science \& Technology},
  volume={10},
  number={2},
  pages={4--4},
  year={2021},
  publisher={The Association for Research in Vision and Ophthalmology}
}
```

### License
This project is licensed under the MIT License. See [LICENSE](https://github.com/danielchyeh/meibo-ML/blob/main/LICENSE) for more details. The parts described below follow their original license.

## Acknowledgements
This is a project based on this [pytorch template](https://github.com/victoresque/pytorch-template). The pytorch template is inspired by the project [Tensorflow-Project-Template](https://github.com/MrGemy95/Tensorflow-Project-Template) by [Mahmoud Gemy](https://github.com/MrGemy95). The template of this github page partially follows [RIDE](https://github.com/frank-xwang/RIDE-LongTailRecognition). The codebase of our work partially follows [NPID](https://github.com/zhirongw/lemniscate.pytorch).



