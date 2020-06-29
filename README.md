# adaptive-inertia-adai

The Pytorch Implementation of Adaptive Inertia Methods. 

The algorithms are based on the paper:  

"Adai: Separating the Effects of Adaptive Learning Rate and Momentum Inertia".


# The environment is as bellow:

Python 3.7.3 

PyTorch >= 1.4.0


# Code Example: 

import adai_optim

#You may use it as a standard Pytorch optimizer.

optimizer = adai_optim.Adai(net.parameters(), lr=lr, betas=(0.1, 0.99), eps=1e-03)


# Performance

Table 1. Test performance comparison. We report the mean and the standard deviations of the optimal test errors and the standard deviation computed over three runs on CIFAR-10 and CIFAR-100, and the optimal test errors over one run on ImageNet.

| Dataset                      | Model       | Adai                      | SGD M            | 
| :--------------------------- | :---------- | :------------------------ | :--------------- |
| CIFAR-10                     | ResNet18    | 4.80  | 4.98  | 
|                              | VGG16       | 6.24  | 6.42  | 
| CIFAR-100                    | DenseNet121 | 19.52 | 19.62 | 
|                              | GoogLeNet   | 20.60 | 21.05 | 
| ImageNet <span>(Top1)        | ResNet50    | 23.20 | 23.51 |
| ImageNet<span>(Top5)         | ResNet50    | 6.62  | 6.82  | 
