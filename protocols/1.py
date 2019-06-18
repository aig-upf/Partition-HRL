data = {
        "agent_file": "agent",
        "agent_name": "AgentQMontezuma",

        "seeds": [0],
        "number_episodes": 3,
        "probability_random_action_agent": 0.1,
        "probability_random_action_option": 0.1,

        "penalty_death_option": -100,
        "penalty_option_action": -1,
        "penalty_option_idle": -0.5,

        "learning_rate": 0.04,
        "reward_end_option": 100,
        "penalty_end_option": -100,

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