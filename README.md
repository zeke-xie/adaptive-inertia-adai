# adaptive-inertia-adai

The Pytorch Implementation of Adaptive Inertia Methods. 


Adaptive Inertia Optimization is proposed in our work:  

[Adaptive Inertia: Disentangling the Effects of Adaptive Learning Rate and Momentum](https://arxiv.org/abs/2006.15815).

This work has been accepted as a **Long Presentation paper (Acceptance Rate ~ 2%) at ICML 2022**.

In this work, we design a novel adaptive optimization method named Adaptive Inertia (Adai), which uses parameter-wise inertia (the momentum hyperparameter as a vector) to accelerate saddle-point escaping and provably select flat minima as well as SGD. Adai combines the advantages of Adam and SGD on saddle-point escaping and minima selection, respectively.

Our experiments demonstrate that Adai can significantly outperform SGD and existing Adam variants for various DNNs where flat minima are desired. 


# The environment is as bellow:

Python 3.7.3 

PyTorch >= 1.4.0


# Usage

You may use it as a standard PyTorch optimizer. 

```python
import adai_optim

optimizer = adai_optim.Adai(net.parameters(), lr=lr, betas=(0.1, 0.99), eps=1e-03, weight_decay=5e-4)
```

# Hyperparameters

The recommended learning rate of Adai is equal to the choice of SGD or 10 times the choice of SGD Momentum (beta=0.9).

The recommended weight decay of Adai is euqal to the choice of SGD and SGD Momentum.

# Test performance



| Dataset   | Model       | AdaiW                    | Adai                     | SGD M                | Adam                 | AMSGrad              | AdamW                | AdaBound             | Padam                | Yogi                 | RAdam                |
|:----------|:------------|:-------------------------|:---------------------|:---------------------|:---------------------|:---------------------|:---------------------|:---------------------|:---------------------|:---------------------|:---------------------|
| CIFAR-10  | ResNet18    | **4.59**<sub>0.16</sub>  | 4.74<sub>0.14</sub>  | 5.01<sub>0.03</sub>  | 6.53<sub>0.03</sub>  | 6.16<sub>0.18</sub>  | 5.08<sub>0.07</sub>  | 5.65<sub>0.08</sub>  | 5.12<sub>0.04</sub>  | 5.87<sub>0.12</sub>  | 6.01<sub>0.10</sub>  |
|           | VGG16       | **5.81**<sub>0.07</sub>  | 6.00<sub>0.09</sub>  | 6.42<sub>0.02</sub>  | 7.31<sub>0.25</sub>  | 7.14<sub>0.14</sub>  | 6.48<sub>0.13</sub>  | 6.76<sub>0.12</sub>  | 6.15<sub>0.06</sub>  | 6.90<sub>0.22</sub>  | 6.56<sub>0.04</sub>  |
| CIFAR-100 | ResNet34    | 21.05<sub>0.10</sub> | **20.79**<sub>0.22</sub> | 21.52<sub>0.37</sub> | 27.16<sub>0.55</sub> | 25.53<sub>0.19</sub> | 22.99<sub>0.40</sub> | 22.87<sub>0.13</sub> | 22.72<sub>0.10</sub> | 23.57<sub>0.12</sub> | 24.41<sub>0.40</sub> |
|           | DenseNet121 | **19.44**<sub>0.21</sub> | 19.59<sub>0.38</sub> | 19.81<sub>0.33</sub> | 25.11<sub>0.15</sub> | 24.43<sub>0.09</sub> | 21.55<sub>0.14</sub> | 22.69<sub>0.15</sub> | 21.10<sub>0.23</sub> | 22.15<sub>0.36</sub> | 22.27<sub>0.22</sub> |
|           | GoogLeNet   | **20.50**<sub>0.25</sub> | 20.55<sub>0.32</sub> | 21.21<sub>0.29</sub> | 26.12<sub>0.33</sub> | 25.53<sub>0.17</sub> | 21.29<sub>0.17</sub> | 23.18<sub>0.31</sub> | 21.82<sub>0.17</sub> | 24.24<sub>0.16</sub> | 22.23<sub>0.15</sub> |
 
# Citing

If you use Adai or other Adai variants in your work, please cite [Adaptive Inertia: Disentangling the Effects of Adaptive Learning Rate and Momentum](https://arxiv.org/abs/2006.15815).

```
@InProceedings{xie2022adaptive,
  title = 	 {Adaptive Inertia: Disentangling the Effects of Adaptive Learning Rate and Momentum},
  author =       {Xie, Zeke and Wang, Xinrui and Zhang, Huishuai and Sato, Issei and Sugiyama, Masashi},
  booktitle = 	 {Proceedings of the 39th International Conference on Machine Learning},
  pages = 	 {24430--24459},
  year = 	 {2022}
  volume = 	 {162},
  series = 	 {Proceedings of Machine Learning Research}
}
```
