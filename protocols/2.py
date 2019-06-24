data = {"agent_name": "AgentDQN",
        "agent_file": "agent_dqn",

        "seeds": [0],
        "number_episodes": 3000,
        "probability_random_action_agent": 0.1,
        "probability_random_action_option": 0.1,

        "epsilon": 0.1,  # exploration rate
        "epsilon_min": 0.0001,
        "epsilon_decay": 0.95,

        "penalty_death_option": -100,
        "penalty_option_action": -1,
        "penalty_option_idle": -0.5,

        "learning_rate": 0.002,
        "reward_end_option": 100,
        "penalty_end_option": -1000,

        # environment's parameters
        "env_name": "MontezumaRevenge-v0",
        "obs_wrapper_name": "obs",

        "NUMBER_ZONES_MONTEZUMA_X": (2 ** 5) * 5,
        "NUMBER_ZONES_MONTEZUMA_Y": 2 * 3 * 5 * 7,

        "NUMBER_ZONES_OPTION_X": (2 ** 3) * 5,
        "NUMBER_ZONES_OPTION_Y": 3 * 7,
        "THRESH_BINARY_OPTION": 0,

        "NUMBER_ZONES_AGENT_X": 2 ** 3,
        "NUMBER_ZONES_AGENT_Y": 7,
        "THRESH_BINARY_AGENT": 40
        }

data.update({"ZONE_SIZE_OPTION_X": data["NUMBER_ZONES_MONTEZUMA_X"] // data["NUMBER_ZONES_OPTION_X"],
             "ZONE_SIZE_OPTION_Y": data["NUMBER_ZONES_MONTEZUMA_Y"] // data["NUMBER_ZONES_OPTION_Y"],
             "ZONE_SIZE_AGENT_X": data["NUMBER_ZONES_MONTEZUMA_X"] // data["NUMBER_ZONES_AGENT_X"],
             "ZONE_SIZE_AGENT_Y": data["NUMBER_ZONES_MONTEZUMA_Y"] // data["NUMBER_ZONES_AGENT_Y"]})