# How to run it ?

clone this repo and go to `protocols/` to select a protocol for your experiment. Let's say you have chosen protocol `7`.
Then execute

```
cd RL
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
python . p
```

# How does it work ?

In this repo you can use/implement your own (Reinforcement Learning algorithms)[https://en.wikipedia.org/wiki/Reinforcement_learning]
using a Hierarchical approach (see for instance this post on (thegradient)[https://thegradient.pub/the-promise-of-hierarchical-reinforcement-learning/].

Hierarchical learning is a method to solve efficiently RL problems, especially when the exploration of your environment is hard.
The main idea is to break down a complex task into smaller ones: "HRL [...] operates on different levels of temporal abstraction".
The manager plans *abstract* macro action and activates *workers* (also called *options*) to accomplish each of them, those interact directly .with the environment.
The manager can also call a special option for exploring the environment.

Within this repo, you can create your own HRL algorithm by inheriting from `AbstractManager` class (see *abstract/manager/manager.py* file).
Then, you just need to code your own manager's policy, the options' policy and the exploring option.
Once this is done, create your own protocol, give it a name, set the relevant parameters and execute `python . name-of-your-protocol` at the root of the project.
In the abstract classes (in folder *abstract*), you will find the method that you have to implement for the manager and the options.

Some examples are available in the folders *baseline*. In *a2c*, we use the *a2c* algorithm for the option and a planning strategy at the top level.
We are open to receive any new manager or options to expand our set of baseline strategies.

# About the environment and the manager's abstract representation

Your environment has to inherit from (gym)[https://gym.openai.com] environment. You can set its name in your protocol's parameters.
Our manager and options use the information given by then environment through the function `step`. This function returns (among others)
the new state of the environment when a certain action is performed from an another state.
You will need to transform these states to make to feed the state list of your agent's policy.
For this purpose, we have created an abstract wrapper class `ObsPixelWrapper` (see also some examples in folder *wrapper*).
You can use this class to, for instance, transform the pixels of your state to make a new gray-scaled-downsampled state.

[![made-with-OpenAIGym](https://img.shields.io/badge/Made%20with-OpenAI%20Gym-1f425f.svg)](https://gym.openai.com/)
[![made-with-Python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![made-with-OpenCV](https://img.shields.io/badge/Made%20with-OpenCV-1f425f.svg)](https://opencv.org/)