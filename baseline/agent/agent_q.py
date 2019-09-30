from baseline.agent.agent import AbstractAgent


class PlainQLearning(AbstractAgent):
    """
    Plan Q Learning
    No options here: this manager is an manager !
    This agent is a baseline to compare the performances with the option setting
    todo
    """

    def reset(self, initial_state):
        pass

    def act(self, train_episode):
        pass

    def update_agent(self, o_r_d_i, action, train_episode):
        pass
