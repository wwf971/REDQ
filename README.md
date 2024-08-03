# Implementation of Randomized Ensembled Double Q-Learning(REDQ) Algorithm


This repository is a simple and readable implementation of REDQ algorithm proposed in ICLR 2021 paper [**Randomized ensembled double q-learning：learning fast without a model**](https://arxiv.org/abs/2101.05982).

Run following command to train:

```
python src/main.py --env_index 1 --algorithm sac --device cuda:1
```

Each experiment will create a folder in output/, where data will be stored.

## Folder structure

- `src/main.py` is entry point of all training tasks. it includes code for:
    - parsing command line argument parsing
    - initializing gym mujoco task
    - creating folder for experiment
    - main loop of training
- `src/redq.py` and `src/sac.py` contains classes that implement redq and sac algorithm, respectively.
- `src/utils.py` contain class of replay buffer, actor, and critic.
- `src/plot-train.ipynb` plots compariason of learning curve in two experiment instance.

## Hyperparameters
- algorithm. redq or sac.
    - command line argument `--algorithm sac`

- task. mujoco task agent will be trained on.
    - command line argument `--env_index 1`
    - 1 for 'hopper', 2 for 'walker', 3 for 'ant', 4 for 'humanoid'.

- $N$. ensemble size. number of Q functions. default: 20.
    - command line argument `--critic_num 20`

- $M$. number of Q functions selected when . default: 2.
    - command line argument `--critic_num_select 2`

- $G$. UTD(update-to-data) ratio. default: 20
    - how many times to train after one interaction with environment.
    - command line argument `--train_num 2`

- $\rho$. target smoothing coefficient. default: 0.005
    - command line argument `--critic_target_update_ratio 0.005`

- replay buffer size. default: 1e6.
    - command line argument `--replay_buffer_size 1e6`

- maximum training step. default: 5e6
    - commdn line argument `--step_train_max 5e6`

<!-- | Symbol      | Description     | Default Value |
| ----------- | --------------- | ------------- |
| $\rho$      | target Q function update rate | 0.005 |
| Paragraph   | Text        | -->

## Experiments

All experimetns are conducted using default hyperparameters values, which are same to those used in REDQ paper.

![hopper](result/train-cmp-hopper.png)
![walker](result/train-cmp-walker.png)
![ant](result/train-cmp-ant.png)
![humanoid](result/train-cmp-humanoid.png)