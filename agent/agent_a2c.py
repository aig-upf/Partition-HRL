from ao.options.options import OptionAbstract
import numpy as np
from agent.agent import AgentQMontezuma
from agent.utils import ExperienceReplay
from agent.models import A2CEager, CriticNetwork, ActorNetwork, SharedConvLayers
import tensorflow as tf
import os


class AgentA2C(AgentQMontezuma):

    def __init__(self, action_space, parameters):
        super().__init__(action_space, parameters)
        # this parameter is useful if you want to restart the episode after n actions (200 actions for instance)
        self.nb_actions = 0

    def reset(self, initial_state):
        print(len(self))
        self.nb_actions = 0
        super().reset(initial_state)

    def check_end_agent(self, o_r_d_i, current_option, train_episode):
        self.nb_actions += 1
        return super().check_end_agent(o_r_d_i, current_option, train_episode) or bool(self.nb_actions > 200)

    def get_option(self) -> OptionAbstract:
        return OptionA2C(self.action_space, self.parameters, len(self), )

    def update_agent(self, o_r_d_i, option, train_episode=None):
        super().update_agent(o_r_d_i, option, train_episode)


class OptionA2C(OptionAbstract):

    def __init__(self, action_space, parameters, index):
        super().__init__(action_space, parameters, index)
        self.input_shape_nn = [None, self.parameters["NUMBER_ZONES_MONTEZUMA_Y"],
                               self.parameters["NUMBER_ZONES_MONTEZUMA_X"], 1]

        self.state_size = self.input_shape_nn[1:]
        self.state_dimension = tuple(self.state_size)

        self.action_size = len(action_space)
        self.observation_model = SharedConvLayers()
        self.a2cDNN = A2CEager(self.input_shape_nn, 32, self.action_size, 'ReinforceNetwork', 'cpu:0', CriticNetwork,
                               ActorNetwork, self.parameters["learning_rate_actor"],
                               self.parameters["learning_rate_critic"], self.observation_model)

        self.device = self.parameters["DEVICE"]
        self.gamma = self.parameters["GAMMA"]
        self.learning_rate_actor = self.parameters["LEARNING_RATE_ACTOR"]
        self.learning_rate_critic = self.parameters["LEARNING_RATE_CRITIC"]
        self.results_folder = self.parameters["RESULTS_FOLDER"]
        self.file_name = self.parameters["FILE_NAME"]
        self.batch_size = self.parameters["BATCH_SIZE"]
        self.weight_ce_exploration = self.parameters["WEIGHT_CE_EXPLORATION"]
        self.buffer = ExperienceReplay(self.batch_size)
        self.main_model_nn = SharedConvLayers()
        self.analyze_memory = analyze_memory

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

        tf.enable_eager_execution()

        # Just to be sure that we don't have some others graph loaded
        tf.reset_default_graph()

        self.sess = None
        self.randomAgent = None

        self.preprocess = Preprocessing(*self.state_dimension)
        self.env = Environment(self.PROBLEM, self.preprocess)

    def _get_actor_critic_error(self, batch):

        no_state = np.zeros(self.state_dimension)

        states_t = np.array([o[1][0] for o in batch])
        states_t1 = np.array([(no_state if o[1][3] is None else o[1][3]) for o in batch])

        p = self.main_model_nn.prediction_critic(states_t)
        p_ = self.main_model_nn.prediction_critic(states_t1)

        x = np.zeros((len(batch),) + self.state_dimension)
        adv_actor = np.zeros(len(batch))
        a_one_hot = np.zeros((len(batch), len(self.action_space)))
        y_critic = np.zeros((len(batch), 1))

        for i in range(len(batch)):
            o = batch[i][1]
            s = o[0]
            a = o[1]
            r = o[2]
            s_ = o[3]

            a_index = self.action_space.index(a)
            if s_ is None:
                t = r
                adv = t - p[i]
            else:
                t = r + self.gamma * p_[i]
                adv = t - p[i]

            x[i] = s
            y_critic[i] = t
            a_one_hot[i][a_index] = 1
            adv_actor[i] = adv

        return x, adv_actor, a_one_hot, y_critic

    def act(self, s):

        predict = self.main_model_nn.prediction_actor([s])[0]

        return np.random.choice(self.action_space, p=predict)

    def observe(self, sample):  # in (s, a, r, s_) format

        self.buffer.add(sample)

    def replay(self):

        if self.buffer.buffer_len() >= self.batch_size:
            batch = self.buffer.sample(self.batch_size, False)

            x, adv_actor, a_one_hot, y_critic = self._get_actor_critic_error(batch)

            self.main_model_nn.train_actor(x, a_one_hot, adv_actor, self.weight_ce_exploration)
            self.main_model_nn.train_critic(x, y_critic)

    def update_option(self, *args, **kwargs):
        """
        the function the updates the option's parameters.
        In your case it could be self.observe but I'm not sure
        :param args:
        :param kwargs:
        :return:
        """

    def reset(self, *args, **kwargs):
        """
        reset the parameter of the option after an episode.
        For instance: the initial and terminal states. Can be any other attribute.
        I expect that you *DO* *NOT* reset the values of the neural networks in this function.
        :param args:
        :param kwargs:
        :return:
        """
        pass

    def compute_total_reward(self, *args, **kwargs):
        """
        processes the information from the environment after a step. Returns a value to update the option
        :param args:
        :param kwargs:
        :return:
        """
        pass

    def compute_total_score(self, *args, **kwargs):
        """
        this function is a metric served in simulate mode. Not so important for the moment
        :param args:
        :param kwargs:
        :return:
        """
        pass
