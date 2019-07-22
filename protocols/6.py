from agent.a2c.models import SharedConvLayers, CriticNetwork, ActorNetwork
import tensorflow as tf
import os
from gym_minigrid.register import env_list



os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

tf.enable_eager_execution()
# todo fix this The name tf.enable_eager_execution is deprecated. Please use tf.compat.v1.enable_eager_execution instead


# Just to be sure that we don't have some others graph loaded
tf.reset_default_graph()
# todo fix this:  The name tf.reset_default_graph is deprecated. Please use tf.compat.v1.reset_default_graph instead.

shared_conv_layers = SharedConvLayers()
critic_network = CriticNetwork(64, shared_conv_layers)
actor_network = ActorNetwork(64, 18, shared_conv_layers)

data = {
        "agent_file": "a2c.agent_a2c",
        "agent_name": "AgentA2C",
        "max_number_actions": 1000,
        "display_environment": True,

        "seeds": [3],
        "number_episodes": 100000,

        "learning_rate": 0.001,

        # please check the values below
        "DEVICE": 'cpu:0',
        "GAMMA_MAX": 0.99,
        "GAMMA_MIN": 0.1,
        "EVOLUTION": "static",
        "LEARNING_RATE": 0.0001,
        "BATCH_SIZE": 6,
        "WEIGHT_CE_EXPLORATION": 0.01,
        "SHARED_CONVOLUTION_LAYERS": shared_conv_layers,
        "CRITIC_NETWORK": CriticNetwork,
        "ACTOR_NETWORK": ActorNetwork,

        # do we need this ?
        "probability_random_action_agent": 0.1,
        "probability_random_action_agent_decay": 1/5000,
        "penalty_death_option": -1,
        "penalty_option_action": -0.1,
        "reward_end_option": 1,
        "penalty_end_option": -1,

        # environment's parameters
        "env_name": "MiniGrid-Empty-5x5-v0",
        "obs_wrapper_name": "minigrid_obs",
        "stack_images_length": 4,
        "OPTION_OBSERVATION_IMAGE_WIDTH": None,
        "OPTION_OBSERVATION_IMAGE_HEIGHT": None,
        "NUMBER_ZONES_GRIDWORLD_X": 608,
        "NUMBER_ZONES_GRIDWORLD_Y": 608,
        "NUMBER_ZONES_OPTION_X": 608,
        "NUMBER_ZONES_OPTION_Y": 608,
        "NUMBER_ZONES_AGENT_X": int(608/32),
        "NUMBER_ZONES_AGENT_Y": int(608/32),
        "THRESH_BINARY_OPTION": 0,
        "THRESH_BINARY_AGENT": 40
        }

data.update({"ZONE_SIZE_AGENT_X": data["NUMBER_ZONES_GRIDWORLD_X"] // data["NUMBER_ZONES_AGENT_X"],
             "ZONE_SIZE_AGENT_Y": data["NUMBER_ZONES_GRIDWORLD_Y"] // data["NUMBER_ZONES_AGENT_Y"]})