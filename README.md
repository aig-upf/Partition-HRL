# How to run it ?

Choose a protocol p (for instance p=7 or 8)

clone this repo and

```
cd RL
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
python . p
```

All the protocol parameters are stored in protocols/p.py.

# How does it work ?

These classes use the library (manager_option_library)[git@github.com:DamienAllonsius/manager_option_library.git] to create agents and options (see an example in folder a2c).

Your agents have to inherit from AbstractManager (see mo/manager/manager.py file) and your options from classes in folder mo/options. You need also a special option for exploring the environment as well (same folder).

Now look at the __main__.py file. The methods `manager.train(...)` and `manager.simulate(...)` applied the option strategy : a manager create an option, make it act, update it and so on (see function `_train_simulate` in manager.py).