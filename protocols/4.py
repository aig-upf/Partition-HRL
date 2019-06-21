from agent.models import SharedConvLayers

shared_conv_layers = SharedConvLayers()




data = {
        "agent_file": "agent_a2c",
        "agent_name": "AgentA2C",

        "seeds": [0],
        "number_episodes": 10000,

        "learning_rate": 0.001,

        # please check the values below
        "DEVICE": 'cpu:0',
        "GAMMA": 0.99,
        "LEARNING_RATE_ACTOR": 0.00001,
        "LEARNING_RATE_CRITIC": 0.0001,
        "BATCH_SIZE": 32,
        "WEIGHT_CE_EXPLORATION": 0.01,
        "SHARED_CONVOLUTION_LAYERS": shared_conv_layers,

        # do we need this ?
        "probability_random_action_agent": 0.1,
        "penalty_death_option": -1,
        "penalty_option_action": -0.1,
        "reward_end_option": 1,
        "penalty_end_option": -1,

        # Montezuma's parameters
        "NUMBER_ZONES_MONTEZUMA_X": (2 ** 5) * 5,
        "NUMBER_ZONES_MONTEZUMA_Y": 2 * 3 * 5 * 7,
        "NUMBER_ZONES_OPTION_X": (2 ** 3) * 5,
        "NUMBER_ZONES_OPTION_Y": 5 * 3 * 7,
        "THRESH_BINARY_OPTION": 0,
        "NUMBER_ZONES_AGENT_X": 2 ** 3,
        "NUMBER_ZONES_AGENT_Y": 7,
        "THRESH_BINARY_AGENT": 40
        }

data.update({"ZONE_SIZE_OPTION_X": data["NUMBER_ZONES_MONTEZUMA_X"] // data["NUMBER_ZONES_OPTION_X"],
             "ZONE_SIZE_OPTION_Y": data["NUMBER_ZONES_MONTEZUMA_Y"] // data["NUMBER_ZONES_OPTION_Y"],
             "ZONE_SIZE_AGENT_X": data["NUMBER_ZONES_MONTEZUMA_X"] // data["NUMBER_ZONES_AGENT_X"],
             "ZONE_SIZE_AGENT_Y": data["NUMBER_ZONES_MONTEZUMA_Y"] // data["NUMBER_ZONES_AGENT_Y"]})