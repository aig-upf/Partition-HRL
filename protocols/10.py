from wrapper.obs import *

data = {"display_environment": True,
        "episodes_performances": 100,
        "number_episodes": 1000,
        "seeds": [0],
        # parameters for the options's policy
        "probability_random_action_option": 0.1,
        "random_decay": 1 / 100,
        "discount_factor": 0.1,
        # parameters for the manager's policy
        "manager_file": "baseline.manager.manager_q_learning",
        "manager_name": "ManagerQLearning",
        "edge_cost": -0.01,
        "max_explore": 10,
        "probability_random_action_manager": 0.1,
        "reward_end_option": 0.1,
        "penalty_end_option": -0.1,
        # parameters for environment's observations (manager and option)
        "env_name": "MiniGrid-Empty-5x5-v0",
        "obs_wrapper_name": Minigrid,
        "stack_images_length": 4,
        "NUMBER_ZONES_GRIDWORLD_X": 160,
        "NUMBER_ZONES_GRIDWORLD_Y": 160,
        "NUMBER_ZONES_MANAGER_X": 5,
        "NUMBER_ZONES_MANAGER_Y": 5,
        "THRESH_BINARY_MANAGER": 58
        }

data.update(
    {"ZONE_SIZE_MANAGER_X": data["NUMBER_ZONES_GRIDWORLD_X"] // data["NUMBER_ZONES_MANAGER_X"],
     "ZONE_SIZE_MANAGER_Y": data["NUMBER_ZONES_GRIDWORLD_Y"] // data["NUMBER_ZONES_MANAGER_Y"]})
