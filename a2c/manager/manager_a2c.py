from a2c.option.option_a2c import A2COption
from mo.options.options_explore import OptionRandomExplore
from mo.manager.manager import AbstractManager
from mo.policies.policy_manager import GraphPlanningPolicyManager


class ManagerA2C(AbstractManager):
    def new_policy(self):
        return GraphPlanningPolicyManager(self.parameters)

    def new_explore_option(self):
        return OptionRandomExplore(self.action_space)

    def new_option(self):
        return A2COption(self.action_space, self.parameters, self.get_number_options())

    def check_end_manager(self, o_r_d_i):
        """
        post-filter
        :param o_r_d_i:
        :return:
        """
        # o_r_d_i[-1]['ale.lives'] != 6
        return super().check_end_manager(o_r_d_i)

    def check_end_option(self, option, o_r_d_i):
        """
        post-processing to monitor the option's performances
        """
        correct_termination = super().check_end_option(option, o_r_d_i)
        if correct_termination:
            print("initial state:", self.policy.get_current_state())
            print("option's mission", self.policy.get_next_state())
            print("current state", o_r_d_i[0]["manager"])

        return correct_termination
