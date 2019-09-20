from a2c.option.option_a2c import A2COption, AbstractOption
from abstract.options.options_explore import OptionRandomExplore
from abstract.manager.manager import AbstractManager
from baseline.policy.graph_planning import GraphPlanningPolicyManager, GraphPseudoCountReward
from abstract.utils.miscellaneous import check_type


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


class ManagerA2CPCIntraRewards(AbstractManager):
    def new_policy(self):
        return GraphPseudoCountReward(self.parameters)

    def new_explore_option(self):
        return OptionRandomExplore(self.action_space)

    def new_option(self):
        return A2COption(self.action_space, self.parameters, self.get_number_options())

    def _train_simulate(self, env, train_episode=None):
        """
        Method used to train or simulate the manager (the main loop)

        a) choose an option
        b) option acts and updates
        c) if a new state is found then update manager

        :param env: the environment.
        :param train_episode: the episode of training.
        - if not None: training
        - if None: simulating
        :return: void
        """
        # The initial observation
        o_r_d_i = [env.reset()] + [None]*3  # o_r_d_i means "Observation_Reward_Done_Info"
        # Reset all the manager parameters
        self.reset(o_r_d_i[0]["manager"])
        done = False
        current_option = None
        # Render the current state
        if self.parameters["display_environment"]:
            self.show_render.render(o_r_d_i[0])

        while not done:
            # If no option is activated then choose one
            if current_option is None:
                current_option = self.select_option(o_r_d_i, train_episode)
                assert current_option.score == 0, "the option's reset function must reset the score to 0."

            # choose an action
            action = current_option.act(train_episode)

            # make an action and display the state space
            o_r_d_i = env.step(action)
            if self.parameters["display_environment"]:
                self.show_render.render(o_r_d_i[0])

            # check if the option ended correctly
            correct_termination = self.check_end_option(current_option, o_r_d_i[0]["manager"])

            # update the option
            intra_reward = self.compute_intra_reward(o_r_d_i, correct_termination)
            current_option.update_option(o_r_d_i, action, correct_termination, train_episode, intra_reward)

            # If the option is done, update the manager
            if correct_termination is not None:
                if check_type(current_option, AbstractOption):
                    # record the correct transition when the option is a regular option (i.e. not an explore option)
                    self.successful_transition.append(correct_termination)
                    self.write_success_rate_transitions()

                # the manager does not need to know if the correct_termination is 0 or 1.
                self.update_manager(o_r_d_i, current_option, train_episode)

                current_option = None

            done = self.check_end_manager(o_r_d_i)

        self.write_manager_score(train_episode)

    def compute_intra_reward(self, o_r_d_i, correct_termination):
        # Lorenzo's remark, this reward should be applied only when correct_termination == True
        number_next_options = self.policy.get_number_next_options(o_r_d_i[0]["manager"])
        if number_next_options == 0 or not correct_termination:
            return 0

        else:
            return max([option.get_critic_value(o_r_d_i[0]["option"]) for
                        option in self.option_list[:number_next_options]])
