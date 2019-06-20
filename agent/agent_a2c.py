from ao.options.options import OptionAbstract
import numpy as np
from agent.agent import AgentQMontezuma
from agent.utils import ExperienceReplay
from agent.models import A2CEager, CriticNetwork, ActorNetwork, SharedConvLayers

class AgentA2C(AgentQMontezuma):

    def __init__(self, action_space, parameters):
        super().__init__(action_space, parameters)
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

        if type(option).__name__ != "OptionRandomExplore":
            if option.terminal_state != o_r_d_i[0]["agent"]:
                print("..")


class OptionA2C(OptionAbstract):

    def __init__(self, action_space, parameters, index):
        super().__init__(action_space, parameters, index)
        self.input_shape_nn = [None, 240, 180, 1]
        self.state_size = [240, 180, 1]
        self.action_size = len(action_space)
        self.observation_model = SharedConvLayers()
        self.a2cDNN = A2CEager(self.input_shape_nn, 32, self.action_size, 'ReinforceNetwork', 'cpu:0', CriticNetwork,
                               ActorNetwork, self.parameters["learning_rate_actor"], self.parameters["learning_rate_critic"],
                               self.observation_model)







    def __init__(self, action_space, parameters, index):
        super().__init__(action_space, parameters, index)
        self.state_size = 1  # gives the hashed image
        self.action_size = len(action_space)
        self.gamma = 0.03    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.state = None

    def act(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        # select random action with prob=epsilon else action=maxQ

        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)

        act_values = self.model.predict(self.state)  # [[action 1, action 2]]
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        # sample random transitions
        choices = np.array(self.memory)
        idx = np.random.choice(len(choices), batch_size)
        minibatch = choices[idx]

        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                Q_next = self.model.predict(next_state)[0]
                target = (reward + self.gamma * np.amax(Q_next))

            target_f = self.model.predict(state)
            target_f[0][action] = target
            # train network
            self.model.fit(state, target_f, epochs=1, verbose=0)

    def update_option(self, o_r_d_i, action, train_episode=None):
        self.memory.append((self.state, action, o_r_d_i[1], o_r_d_i[0]["option"], o_r_d_i[2]))
        self.state = np.array([o_r_d_i[0]["option"]])

    def reset(self, initial_state, current_state, terminal_state):
        super().reset_states(initial_state, terminal_state)
        self.state = np.array([current_state])

    def compute_total_score(self, *args, **kwargs):
        pass

    def compute_total_reward(self, o_r_d_i, action, end_option):
        """
        test ok
        :param o_r_d_i:
        :param action:
        :param end_option:
        :return:
        """
        total_reward = super().compute_total_reward(o_r_d_i, action, end_option)
        total_reward += (action != 0) * self.parameters["penalty_option_action"]
        total_reward += (action == 0) * self.parameters["penalty_option_idle"]
        total_reward += o_r_d_i[2] * self.parameters["penalty_death_option"]

        return total_reward

    class A2CAgent:

        np.random.seed(1)

        def __init__(self, action_space, state_dimension, buffer, main_model_nn, gamma, batch_size,
                     weight_ce_exploration, analyze_memory=False):

            self.batch_size = batch_size
            self.buffer = buffer
            self.action_space = action_space
            self.state_dimension = state_dimension
            self.main_model_nn = main_model_nn
            self.analyze_memory = analyze_memory
            self.gamma = gamma
            self.weight_ce_exploration = weight_ce_exploration

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

    class VgridenvEagerAC():

        def __init__(self):
            tf.enable_eager_execution()

            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

            self.IMAGE_WIDTH = 84
            self.IMAGE_HEIGHT = 84
            self.IMAGE_DEPTH = 3
            self.PROBLEM = 'GE_MazeOptions-v1'
            self.DEVICE = 'cpu:0'
            self.ACTION_SPACE = [0, 1, 2, 3]  # [0, 1]
            self.GAMMA = 0.99
            self.LEARNING_RATE_ACTOR = 0.00001
            self.LEARNING_RATE_CRITIC = 0.0001
            self.RESULTS_FOLDER = './results/'
            self.FILE_NAME = 'testMazeOptionsAC.pkl'
            self.BATCH_SIZE = 32
            self.WEIGHT_CE_EXPLORATION = 0.01

            self.preprocess = Preprocessing(self.IMAGE_HEIGHT, self.IMAGE_WIDTH, self.IMAGE_DEPTH)

            self.buffer_memory = ExperienceReplay(self.BATCH_SIZE)

            self.env = Environment(self.PROBLEM, self.preprocess)

            self.input_shape_nn = [None, self.IMAGE_WIDTH, self.IMAGE_HEIGHT, self.IMAGE_DEPTH]  # [None, 4]
            self.stateDimension = (self.IMAGE_WIDTH, self.IMAGE_HEIGHT, self.IMAGE_DEPTH)  # (4,)

            # Just to be sure that we don't have some others graph loaded
            tf.reset_default_graph()

            self.shared_observation_model = SharedConvLayers()
            self.a2cDNN = A2CEager(self.input_shape_nn, 32, len(self.ACTION_SPACE), 'ReinforceNetwork', self.DEVICE,
                                   CriticNetwork, ActorNetwork, self.LEARNING_RATE_ACTOR, self.LEARNING_RATE_CRITIC,
                                   self.shared_observation_model)

            self.sess = None

            self.randomAgent = None

            self.agent = A2CAgent(self.ACTION_SPACE, self.stateDimension, self.buffer_memory, self.a2cDNN, self.GAMMA,
                                  self.BATCH_SIZE, self.WEIGHT_CE_EXPLORATION)
