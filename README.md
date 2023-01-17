# AdapNet:  Adaptive  Semantic  Segmentation in  Adverse  Environmental  Conditions
AdapNet is a deep learning model used for semantic image segmentation, aiming to assign semantic labels (such as car, road, tree, etc.) to every pixel in the input image. It's easily trainable on a single GPU with 12 GB of memory with a quick inference time. It has been benchmarked on Cityscapes, Synthia, ScanNet, SUN RGB-D, and Freiburg Forest datasets.

This repository facilitates the TensorFlow implementation of AdapNet, enabling you to train your model on any dataset and evaluate the results in terms of the mean IoU metric.

It can be further integrated with the [CMoDE](https://github.com/DeepSceneSeg/CMoDE) fusion scheme for multimodal semantic segmentation.

If you find this useful for your research, please consider citing our paper:
```
@inproceedings{valada2017icra,
  title = {AdapNet: Adaptive Semantic Segmentation in Adverse Environmental Conditions},
  booktitle = {Proceedings of the IEEE International Conference on Robotics and Automation (ICRA)},
  pages={4644--4651},
  year = {2017},
  organization={IEEE}
}
```

## Live Demo
http://deepscene.cs.uni-freiburg.de

For any inquiries, please check the Contacts section in this repository.

## System Requirements

Programming Language
Python 2.7

Python Packages
tensorflow-gpu 1.4.0

Check out the Configure the Network section for further details on setting up the project.

## Training and Evaluation
Training and evaluation procedures can be found in the r