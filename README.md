# LORD
Implementation of the method described in the paper: [Latent Optimization for Non-adversarial Representation Disentanglement](http://www.vision.huji.ac.il/lord) by Aviv Gabbay and Yedid Hoshen.

## Pose transfer between identities
| Cars3D | KTH |
| :---: | :---: |
| ![image](http://www.vision.huji.ac.il/lord/img/cars3d/ours.png) | ![image](http://www.vision.huji.ac.il/lord/img/kth/ours.png) |

| SmallNorb-Poses | SmallNorb |
| :---: | :---: |
| ![image](http://www.vision.huji.ac.il/lord/img/smallnorb-poses/ours.png) | ![image](http://www.vision.huji.ac.il/lord/img/smallnorb/ours.png) |

| CelebA |
| :---: |
| ![image](http://www.vision.huji.ac.il/lord/img/celeba/ours.png) |


## Usage
### Dependencies
* python >= 3.6
* numpy >= 1.15.4
* tensorflow-gpu >= 1.12.0
* keras >= 2.2.4
* opencv >= 3.4.4

### Getting started
Training a model for disentanglement requires several steps.

#### Preprocessing an image dataset
Preprocessing a local copy of one of the supported datasets can be done as follows:
```
lord.py --base-dir <output-root-dir> preprocess
    --dataset-id {mnist,smallnorb,cars3d,shapes3d,celeba,kth,edges2shoes}
    --dataset-path <input-dataset-path>
    --data-name <output-data-filename>
```

Splitting a preprocessed dataset into train and test sets can be done according to one of two configurations:
```
lord.py --base-dir <output-root-dir> split-identities
    --input-data-name <input-data-filename>
    --train-data-name <output-train-data-filename>
    --test-data-name <output-test-data-filename>
    --num-test-identities <number-of-random-test-identities>
```

```
lord.py --base-dir <output-root-dir> split-samples
    --input-data-name <input-data-filename>
    --train-data-name <output-train-data-filename>
    --test-data-name <output-test-data-filename>
    --test-split <ratio-of-random-test-samples>
```

#### Training a model
Given a preprocessed train set, training a GLO based model (first stage) can be done as follows:
```
lord.py --base-dir <output-root-dir> train
    --data-name <input-preprocessed-data-filename>
    --model-name <output-model-name>
    --pose-dim <pose-code-size>
    --identity-dim <identity-code-size>
```

The second stage should be trained afterwards as follows:
```
lord.py --base-dir <output-root-dir> train-encoders
    --data-name <input-preprocessed-data-filename>
    --model-name <input-model-name>
```

#### Testing a model
A trained model can be tested similarly:
```
lord.py --base-dir <output-root-dir> test
    --data-name <input-preprocessed-data-filename>
    --model-name <input-model-name>
    --num-samples <number-of-random-samples-for-pose-transfer>
```

## Citing
If you find this project useful for your research, please cite
```
@article{gabbay2019lord,
  author    = {Aviv Gabbay and Yedid Hoshen},
  title     = {Latent Optimization for Non-adversarial Representation Disentanglement},
  journal   = {arXiv preprint arXiv:1906.11796},
  year      = {2019}
}
```
