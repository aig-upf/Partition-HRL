from abstract.utils.key_codes import Key

data = {
        # environment's parameters
        "env_name": "GE_MazeKeyDoorXL-v0",
        "obs_wrapper_name": "obs_pixels_stacked_frames",
        "display_environment": True,
        "stack_images_length": 4,

        "Keyboard_keys": [Key.NUM_0, Key.UP, Key.RIGHT, Key.DOWN, Key.LEFT],

        "OPTION_OBSERVATION_IMAGE_WIDTH": None,
        "OPTION_OBSERVATION_IMAGE_HEIGHT": None,
        "NUMBER_ZONES_GRIDWORLD_X": 84,
        "NUMBER_ZONES_GRIDWORLD_Y": 84,
        "NUMBER_ZONES_OPTION_X": 84,
        "NUMBER_ZONES_OPTION_Y": 84,
        "NUMBER_ZONES_MANAGER_X": 4,
        "NUMBER_ZONES_MANAGER_Y": 4,
        "THRESH_BINARY_OPTION": 0,
        "THRESH_BINARY_MANAGER": 10
        }

data.update({"ZONE_SIZE_MANAGER_X": data["NUMBER_ZONES_GRIDWORLD_X"] // data["NUMBER_ZONES_MANAGER_X"],
             "ZONE_SIZE_MANAGER_Y": data["NUMBER_ZONES_GRIDWORLD_Y"] // data["NUMBER_ZONES_MANAGER_Y"]})
