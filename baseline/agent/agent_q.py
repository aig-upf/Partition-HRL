import numpy as np
from baseline.agent.agent import AbstractAgent


class PlainQLearning(AbstractAgent):
    """
    Plan Q Learning implementation.
    *NOT TESTED*
    No options here: this manager is an manager !
    This agent is a baseline to compare the performances with the option setting
    """

    def __init__(self, action_space, parameters):
        super().__init__(action_space, parameters)
        self.policy = PolicyOptionQArray(action_space, parameters)

    def reset(self, initial_state):
        self.policy.reset(initial_state)

    def act(self, train_episode):
        if (train_episode is not None) and (np.random.rand() < self.parameters["probability_random_action_agent"]):
            return self.make_random_action()

        else:
            return self.policy.find_best_action(train_episode)

    def update_agent(self, o_r_d_i, action, train_episode):
        # update the policy
        total_reward = self.compute_total_reward(o_r_d_i)
        self.policy.update_policy(o_r_d_i[0], total_reward, action, False, train_episode)

        # update the score
        self.score += self.compute_total_score(o_r_d_i)
