from a2c.utils.models import SharedConvLayers, CriticNetwork, ActorNetwork
import tensorflow as tf
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

tf.logging.set_verbosity(tf.logging.ERROR)

tf.enable_eager_execution()
# todo fix this The name tf.enable_eager_execution is deprecated. Please use tf.compat.v1.enable_eager_execution instead

# Just to be sure that we don't have some others graph loaded
tf.reset_default_graph()
# todo fix this:  The name tf.reset_default_graph is deprecated. Please use tf.compat.v1.reset_default_graph instead.

shared_conv_layers = SharedConvLayers()

data = {
        "manager_file": "a2c.manager.manager_a2c",
        "manager_name": "ManagerA2CPCIntraRewards",
        "max_number_actions": 1000,
        "verbose": False,
        "episodes_performances": 200,
        "seeds": range(5),
        "number_episodes": 50000,
        "learning_rate": 0.001,

        # Policy Options
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
        "reward_end_option": 0.6,
        "penalty_end_option": -0.6,
        "intra_reward_coefficient": 0.1,

        # policy manager
        "edge_cost": -0.01,
        "max_explore": 10,
        "probability_random_action_manager": 0.1,

        # environment's parameters
        "env_name": "GE_MazeKeyDoorXL-v0",
        "obs_wrapper_name": "obs_pixels_stacked_frames",
        "display_environment": True,
        "stack_images_length": 4,
        "OPTION_OBSERVATION_IMAGE_WIDTH": None,
        "OPTION_OBSERVATION_IMAGE_HEIGHT": None,
        "NUMBER_ZONES_GRIDWORLD_X": 84,
        "NUMBER_ZONES_GRIDWORLD_Y": 84,
        "NUMBER_ZONES_OPTION_X": 84,
        "NUMBER_ZONES_OPTION_Y": 84,
        "NUMBER_ZONES_MANAGER_X": 4,
        "NUMBER_ZONES_MANAGER_Y": 4,
        "THRESH_BINARY_OPTION": 0,
        "THRESH_BINARY_MANAGER": 10,

        "GRAY_SCALE": False,
        "SSIM_PRECISION_FACTOR": 3
        }

data.update({"ZONE_SIZE_MANAGER_X": data["NUMBER_ZONES_GRIDWORLD_X"] // data["NUMBER_ZONES_MANAGER_X"],
             "ZONE_SIZE_MANAGER_Y": data["NUMBER_ZONES_GRIDWORLD_Y"] // data["NUMBER_ZONES_MANAGER_Y"]})
