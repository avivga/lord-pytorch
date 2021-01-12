# LORD
> [Demystifying Inter-Class Disentanglement](http://www.vision.huji.ac.il/lord)  
> Aviv Gabbay and Yedid Hoshen  
> International Conference on Learning Representations (ICLR), 2020.  
> Pytorch re-implementation (thanks to [@dneuhof](https://github.com/dneuhof)) [[Official tensorflow implementation](https://github.com/avivga/lord)]

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
* pytorch >= 1.3.0
* opencv >= 3.4.4
* dlib >= 19.17.0

### Getting started
Training a model for disentanglement requires several steps.

#### Preprocessing an image dataset
Preprocessing a local copy of one of the supported datasets can be done as follows:
```
lord.py --base-dir <output-root-dir> preprocess
    --dataset-id {mnist,smallnorb,cars3d,shapes3d,celeba,kth,rafd}
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
@inproceedings{gabbay2020lord,
  author    = {Aviv Gabbay and Yedid Hoshen},
  title     = {Demystifying Inter-Class Disentanglement},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2020}
}
```
