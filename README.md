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
