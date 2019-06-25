from ao.policies.agent.agent_policy import PolicyAbstractAgent
from ao.structures.tree import Tree, Node


class QTree(PolicyAbstractAgent):
    """
    This class is used when the number of actions and the number of states are unknown.
    """

    def __init__(self, parameters):
        self.parameters = parameters
        self.tree = None

    def __len__(self):
        return len(self.tree)

    def __str__(self):
        return str(self.tree)

    def reset(self, initial_state):
        if self.tree is None:
            self.tree = Tree(initial_state)

        if self.tree.root.data != initial_state:
            raise ValueError("The initial observation is not always the same ! " +
                             "Maybe a Graph structure could be more adapted")
        else:
            self.tree.current_node = self.tree.root

    def find_best_action(self, state):
        """
        returns the best value and the best action. If best action is None, then explore.
        :return: best_option_index, terminal_state
        """
        values = self.tree.get_values_children()
        if not values:
            return None, None

        if all(val == values[0] for val in values):  # all the values are the same

            # choose a random action with probability distribution computed from the leaves' depths
            best_option_index = self.tree.get_random_child_index()

        else:  # there exists a proper best action
            best_reward = max(values)
            best_option_index = values.index(best_reward)

        return best_option_index, self.tree.get_current_node().children[best_option_index].data

    def update_policy(self, action, reward, new_state):
        """
        Performs the Q learning update :
        Q_{t+1}(current_position, action) = (1- learning_rate) * Q_t(current_position, action)
                                         += learning_rate * [reward + max_{actions} Q_(new_position, action)]

        """
        # node which value attribute is Q_t(current_position, action)
        node_activated = self.tree.get_child_node(action)

        try:
            new_node = self.tree._get_node_from_state(new_state)  # maybe different than node_activated
            if new_node.children:  # there are children, take the maximum value
                best_value = max(new_node.get_values())

            else:  # there are no children -> best_value is 0
                best_value = 0

        except ValueError:  # this new_state does not exist for the moment
            best_value = 0
            self.tree._update(Node(new_state))

        node_activated.value *= (1 - self.parameters["learning_rate"])
        node_activated.value += self.parameters["learning_rate"] * (reward + best_value)

    def get_max_number_successors(self):
        return self.tree.get_max_width()

    def get_current_state(self):
        return self.tree.get_current_node().data

    def _no_return_update(self, new_state):
        """
        (no return option)
           does not add anything if
           for action in q[option.terminal_state]:
           action.terminal_state = option.initial_state
        """
        try:
            new_node = self.tree._get_node_from_state(new_state)
            for node in new_node.children:
                if node.data == self.tree.current_node.data:
                    return False
            return True

        except ValueError:
            return True
