from agent.a2c.models import SharedConvLayers, CriticNetwork, ActorNetwork
import tensorflow as tf
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

tf.enable_eager_execution()
# todo fix this The name tf.enable_eager_execution is deprecated. Please use tf.compat.v1.enable_eager_execution instead


# Just to be sure that we don't have some others graph loaded
tf.reset_default_graph()
# todo fix this:  The name tf.reset_default_graph is deprecated. Please use tf.compat.v1.reset_default_graph instead.

shared_conv_layers = SharedConvLayers()
critic_network = CriticNetwork(32, shared_conv_layers)

data = {
        "agent_file": "a2c.agent_a2c",
        "agent_name": "AgentA2C",
        "max_number_actions": 1000,

        "seeds": [3],
        "number_episodes": 100000,

        "learning_rate": 0.001,

        # please check the values below
        "DEVICE": 'cpu:0',
        "GAMMA": 0.99,
        "LEARNING_RATE_ACTOR": 0.00001,
        "LEARNING_RATE_CRITIC": 0.0001,
        "BATCH_SIZE": 32,
        "WEIGHT_CE_EXPLORATION": 0.01,
        "SHARED_CONVOLUTION_LAYERS": shared_conv_layers,
        "CRITIC_NETWORK": critic_network,
        "ACTOR_NETWORK": ActorNetwork,

        # do we need this ?
        "probability_random_action_agent": 0.1,
        "penalty_death_option": -1,
        "penalty_option_action": -0.1,
        "reward_end_option": 1,
        "penalty_end_option": -1,

        # environment's parameters
        "env_name": "MontezumaRevenge-v0",
        "obs_wrapper_name": "obs_a2c",
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