# adaptive-inertia-adai

The Pytorch Implementation of Adaptive Inertia Methods. 

The algorithms are based on the paper:  

"Adai: Separating the Effects of Adaptive Learning Rate and Momentum Inertia".
https://arxiv.org/abs/2006.15815


# The environment is as bellow:

Python 3.7.3 

PyTorch >= 1.4.0


# Code Example: 

#You may use it as a standard PyTorch optimizer.

import adai_optim

optimizer = adai_optim.Adai(net.parameters(), lr=lr, betas=(0.1, 0.99), eps=1e-03)


# Performance

Table 1. Test performance comparison. We report the mean of the optimal test errors computed over three runs on CIFAR-10 and CIFAR-100, and the optimal test errors over one run on ImageNet. The table and settings are adopted from the original paper https://arxiv.org/abs/2006.15815.

| Dataset                      | Model       | Adai                      | SGD Momentum| Adai$^{\star}$ | Adam | AMSGrad | AdamW|
| :--------------------------- | :---------- | :------------------------ | :--------------- | :------------------------ | :--------------- | :--------------- | :--------------- |
| CIFAR-10                     | ResNet18    | 4.80  | 4.98  | 5.53 | 6.46 | 6.75 | 6.59|
|                              | VGG16       | 6.24  | 6.42  | 6.80 | 7.85 |8.05 | 7.55
| CIFAR-100                    | DenseNet121 | 19.52 | 19.62 | 21.87 | 25.36 | 25.52 | 25.05 |
|                              | GoogLeNet   | 20.60 | 21.05 | 22.84 | 26.63 | 27.49 | 26.24 |
| ImageNet<span>(Top1)         | ResNet50    | 23.20 | 23.51 | 27.09 | 27.13 | 28.08 | 27.47 |
| ImageNet<span>(Top5)         | ResNet50    | 6.62  | 6.82  | 8.89 | 9.18 | 9.48 | 9.29 |
