# A TensorFlow implementation of reinforcement learning Q-table and Q-network.

### About

`q_table.py` uses reinforcement learning to generate a Q-table specific to the environment in which it is learned.

`q_network.py` uses reinforcement learning to train a neural network that maps an environment state to an action-value tensor output.

### Dependencies

```
gym 0.12.4
numpy 1.16.3
tensorflow (tensorflow-gpu) 1.13.1
six 1.12.0
```

### Instructions

Clone the repository

`git clone https://github.com/Jaewan-Yun/q_learning_tutorial`

Navigate to folder

`cd q_learning_tutorial`

Run desired tutorial code

`python q_table.py`
`python q_network.py`
