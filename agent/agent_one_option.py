from agent.agent import AgentQMontezuma


class AgentOneOption(AgentQMontezuma):
    pass
    """
    def act(self, o_r_d_i, train_episode=None):
        option = super().act(o_r_d_i, train_episode)
        if type(option).__name__ == "Option":
            print(self.option_list[0])
            return self.option_list[0]
        else:
            return option

    def check_end_agent(self, o_r_d_i, current_option, train_episode):
        return current_option.check_end_option(o_r_d_i[0]["agent"])
    """