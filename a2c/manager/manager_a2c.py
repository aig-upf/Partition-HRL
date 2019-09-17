from a2c.option.option_a2c import A2COption
from mo.options.options_explore import OptionRandomExplore
from mo.manager.manager import AbstractManager
from baseline.policy.graph_planning import GraphPlanningPolicyManager, GraphPseudoCountReward


class ManagerA2C(AbstractManager):
    def new_policy(self):
        return GraphPlanningPolicyManager(self.parameters)

    def new_explore_option(self):
        return OptionRandomExplore(self.action_space)

    def new_option(self):
        return A2COption(self.action_space, self.parameters, self.get_number_options())


class ManagerA2CPseudoCount(AbstractManager):
    def new_policy(self):
        return GraphPseudoCountReward(self.parameters)

    def new_explore_option(self):
        return OptionRandomExplore(self.action_space)

    def new_option(self):
        return A2COption(self.action_space, self.parameters, self.get_number_options())
