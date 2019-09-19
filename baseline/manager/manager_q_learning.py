from abstract.manager.manager import AbstractManager
from abstract.options.options_explore import OptionRandomExplore
from baseline.option.option_q_learning import OptionQLearning
from baseline.policy.graph_planning import GraphPlanningPolicyManager


class ManagerQLearning(AbstractManager):

    def new_option(self):
        return OptionQLearning(self.action_space, self.parameters, self.get_number_options())

    def new_explore_option(self):
        return OptionRandomExplore(self.action_space)

    def new_policy(self):
        return GraphPlanningPolicyManager(self.parameters)
