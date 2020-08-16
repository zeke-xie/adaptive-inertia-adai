# adaptive-inertia-adai

The Pytorch Implementation of Adaptive Inertia Methods. 

The algorithms are based on the paper:  

[Adai: Separating the Effects of Adaptive Learning Rate and Momentum Inertia.](https://arxiv.org/abs/2006.15815)


# The environment is as bellow:

Python 3.7.3 

PyTorch >= 1.4.0


# Usage

#You may use it as a standard PyTorch optimizer.

```python
import adai_optim

optimizer = adai_optim.Adai(net.parameters(), lr=lr, betas=(0.1, 0.99), eps=1e-03)
```

  
 
# Citing

If you use Adai or other Adai variants in your work, please cite [Adai: Separating the Effects of Adaptive Learning Rate and Momentum Inertia.](https://arxiv.org/abs/2006.15815).

```
@article{xie2020adai,
  title={Adai: Separating the Effects of Adaptive Learning Rate and Momentum Inertia},
  author={Xie, Zeke and Wang, Xinrui and Zhang, Huishuai and Sato, Issei and Sugiyama, Masashi},
  journal={arXiv preprint arXiv:2006.15815},
  year={2020}
}
```
