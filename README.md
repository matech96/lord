# LORD
Implementation of the method described in the paper: [Demystifying Inter-Class Disentanglement](http://www.vision.huji.ac.il/lord) by Aviv Gabbay and Yedid Hoshen.

## Content transfer between classes
| Cars3D | SmallNorb | KTH |
| :---: | :---: | :---: |
| ![image](http://www.vision.huji.ac.il/lord/img/cars3d/ours.jpg) | ![image](http://www.vision.huji.ac.il/lord/img/smallnorb-poses/ours.png) | ![image](http://www.vision.huji.ac.il/lord/img/kth/ours.png) |

| CelebA |
| :---: |
| ![image](http://www.vision.huji.ac.il/lord/img/celeba/ours.png) |


## Usage
### Dependencies
* python >= 3.6
* numpy >= 1.15.4
* tensorflow-gpu >= 1.12.0
* keras >= 2.2.4
* keras-lr-multiplier >= 0.7.0
* opencv >= 3.4.4
* dlib >= 19.17.0

### Getting started
Training a model for disentanglement requires several steps.

#### Preprocessing an image dataset
Preprocessing a local copy of one of the supported datasets can be done as follows:
```
lord.py --base-dir <output-root-dir> preprocess
    --dataset-id {mnist,smallnorb,cars3d,shapes3d,celeba,kth,rafd,edges2shoes}
    --dataset-path <input-dataset-path>
    --data-name <output-data-filename>
```

Splitting a preprocessed dataset into train and test sets can be done according to one of two configurations:
```
lord.py --base-dir <output-root-dir> split-classes
    --input-data-name <input-data-filename>
    --train-data-name <output-train-data-filename>
    --test-data-name <output-test-data-filename>
    --num-test-classes <number-of-random-test-classes>
```

```
lord.py --base-dir <output-root-dir> split-samples
    --input-data-name <input-data-filename>
    --train-data-name <output-train-data-filename>
    --test-data-name <output-test-data-filename>
    --test-split <ratio-of-random-test-samples>
```

#### Training a model
Given a preprocessed train set, training a model with latent optimization (first stage) can be done as follows:
```
lord.py --base-dir <output-root-dir> train
    --data-name <input-preprocessed-data-filename>
    --model-name <output-model-name>
    --content-dim <content-code-size>
    --class-dim <class-code-size>
```

Training encoders for amortized inference (second stage) can be done as follows:
```
lord.py --base-dir <output-root-dir> train-encoders
    --data-name <input-preprocessed-data-filename>
    --model-name <input-model-name>
```

## Citing
If you find this project useful for your research, please cite
```
@article{gabbay2019lord,
  author    = {Aviv Gabbay and Yedid Hoshen},
  title     = {Demystifying Inter-Class Disentanglement},
  journal   = {arXiv preprint arXiv:1906.11796},
  year      = {2019}
}
```
