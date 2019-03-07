# Bayes by backprop

This repo consists of basic version of Bayes by backprop. 

## Getting Started
I would suggest to use a seperate virtual environment for this purpose so that it doesnot conflict.

### Virtual environment setup

```
virtualenv --python=python3.6 bbb
source bbb/bin/activate
```

### Installation steps
```
git clone --depth=1 https://github.com/luke-97/BayesByBackprop.git
pip install -r requirements.txt
```

## Running
If you are using Wandb use the key to login
```
wandb login <key(in wandb website)>
```
```
cd bbb
python test.py

```

