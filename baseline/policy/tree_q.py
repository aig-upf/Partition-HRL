from mo.utils.miscellaneous import obs_equal
import numpy as np
from mo.manager.manager import AbstractPolicyManager
from mo.structures.tree import Tree


class TreeQPolicyManager(AbstractPolicyManager):
    """
    todo refactor this -> this class is not maintained
    This class is used when the number of actions and the number of states are unknown.
    """

    def get_next_state(self, index_next_state):
        return self.tree.nodes[index_next_state]

    def __init__(self, parameters):
        self.parameters = parameters
        self.tree = None
        self.end_novelty = False

    def __len__(self):
        return len(self.tree)

    def __str__(self):
        return str(self.tree)

    def reset(self, initial_state):
        self.end_novelty = False
        if self.tree is None:
            self.tree = Tree(initial_state)

        if not obs_equal(self.tree.root.data, initial_state):
            raise ValueError("The initial observation is not always the same ! " +
                             "Maybe a Graph structure could be more adapted")
        else:
            self.tree.reset()

    def find_best_action(self, train_episode=None):
        """
        todo take into account train_episode
        returns the best value and the best action. If best action is None, then explore.
        :return: best_option_index, terminal_state
        """
        values = self.tree.get_children_values()
        if train_episode and (not values or np.random.rand() < self.parameters["probability_random_action_agent"] *
                              np.exp(-train_episode * self.parameters["probability_random_action_agent_decay"])):
            return None, None

        if all(val == values[0] for val in values):  # all the values are the same

            # choose a random action with probability distribution computed from the leaves' depths
            best_option_index = self.tree.get_random_child_index()

        else:  # there exists a proper best action
            best_reward = max(values)
            best_option_index = values.index(best_reward)

        return best_option_index, self.tree.get_child_data_from_index(best_option_index)

    def update_policy(self, new_state, reward, action, train_episode=None):
        """
        todo take into account train_episode
        Performs the Q learning update :
        Q_{t+1}(current_position, action) = (1- learning_rate) * Q_t(current_position, action)
                                         += learning_rate * [reward + max_{actions} Q_(new_position, action)]

        and update the current_node position of variable self.tree
        """
        if train_episode is None:
            raise NotImplementedError()

        else:
            if action is None:  # we update from an exploration
                found = self.tree.move_if_node_with_state(new_state)
                if not found:
                    self.end_novelty = self.tree.add_node(new_state)

            else:  # we update from a regular option

                # move to node which value attribute is Q_t(current_position, action)
                self.tree.move_to_child_node_from_index(action)
                node_activated = self.tree.current_node

                # if the action performed well, take the best value
                if obs_equal(node_activated.data, new_state):

                    if node_activated.get_children_values():
                        best_value = max(node_activated.get_children_values())
                    else:
                        best_value = 0

                    node_activated.value *= (1 - self.parameters["learning_rate"])
                    node_activated.value += self.parameters["learning_rate"] * (reward + best_value)

                else:  # set best value to zero, add the node if the new_state is new and update tree.current_state
                    found = self.tree.move_if_node_with_state(new_state)
                    if not found:
                        self.tree.add_node(new_state)

    def get_max_number_successors(self):
        return self.tree.get_max_width()

    def get_current_state(self):
        return self.tree.get_current_state()
