data = {"manager_name": "PlainDQNMontezuma",
        "manager_file": "manager_plain_dqn",
        "display_environment": True,

        "seeds": [0],
        "number_episodes": 200,
        "penalty_lost_life_for_manager": -10,

        "memory": 2000,
        "gamma": 0.03,
        "epsilon": 1.0,
        "epsilon_min": 0.01,
        "epsilon_decay": 0.995,
        "learning_rate": 0.001,

        "env_name": "MontezumaRevenge-v0"
        }
