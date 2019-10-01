from abstract.utils.key_codes import Key

data = {
        # environment's parameters
        "env_name": "MontezumaRevenge-v0",
        "obs_wrapper_name": "obs_pixels_stacked_frames",
        "stack_images_length": 4,

        #  Key.SPACE already taken to change visualization

        "Keyboard_keys": [Key.NUM_1, Key.Z, Key.UP, Key.RIGHT, Key.LEFT, Key.DOWN, Key.NUM_3, Key.NUM_4, Key.NUM_5, Key.NUM_6, None, Key.NUM_7, Key.X, Key.NUM_8, Key.C, Key.NUM_9],

        "NUMBER_ZONES_ENV_X": (2 ** 5) * 5,
        "NUMBER_ZONES_ENV_Y": 2 * 3 * 5 * 7,
        "NUMBER_ZONES_OPTION_X": (2 ** 5) * 5,
        "NUMBER_ZONES_OPTION_Y": 2 * 3 * 5 * 7,
        "THRESH_BINARY_OPTION": 0,
        "NUMBER_ZONES_MANAGER_X": 2 ** 3,
        "NUMBER_ZONES_MANAGER_Y": 3 * 7,
        "THRESH_BINARY_MANAGER": 0
        }

data.update({"ZONE_SIZE_MANAGER_X": data["NUMBER_ZONES_ENV_X"] // data["NUMBER_ZONES_MANAGER_X"],
             "ZONE_SIZE_MANAGER_Y": data["NUMBER_ZONES_ENV_Y"] // data["NUMBER_ZONES_MANAGER_Y"]})