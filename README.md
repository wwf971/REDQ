# Implementation of Randomized Ensembled Double Q-Learning(REDQ) Algorithm


This repository is a simple and readable implementation of REDQ algorithm proposed in ICLR 2021 paper [**Randomized ensembled double q-learning：learning fast without a model**](https://arxiv.org/abs/2101.05982).

Run following command to train:

```
python src/main.py --env_index 1 --algorithm sac --device cuda:1
```

-  env_index: 1 for 'hopper', 2 for 'walker', 3 for 'ant', 4 for 'humanoid'.
- alorithm: sac or redq

Each experiment will create a folder in output/, where data will be stored.