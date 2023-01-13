# AdapNet:  Adaptive  Semantic  Segmentation in  Adverse  Environmental  Conditions
AdapNet is a deep learning model used for semantic image segmentation, aiming to assign semantic labels (such as car, road, tree, etc.) to every pixel in the input image. It's easily trainable on a single GPU with 12 GB of memory with a quick inference time. It has been benchmarked on Cityscapes, Synthia, ScanNet, SUN RGB-D, and Freiburg Forest datasets.

This repository facilitates the TensorFlow implementation of AdapNet, enabling you to train your model on any dataset and evaluate the results in terms of the mean IoU metric.

It can be further integrated with the [CMoDE](https://github.com/DeepSceneSeg/CMoDE) fusion scheme for multimodal semantic segmentation.

If you find this useful for your research, please consider citing our paper:
```
@inproceedings{valada2017icra,
  title = {AdapNet: Adaptive Semantic Segmentation in Adverse Environmental Conditions},
  booktitle = {Proceedings of the IEEE International Con