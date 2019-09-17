from mo.utils.miscellaneous import red, white, find_element_in_list
import numpy as np
from mo.manager.manager import AbstractPolicyManager


class GraphPlanningPolicyManager(AbstractPolicyManager):
    """
    A policy for the manager based on a graph.
    Number of states unknown.
    Number of actions unknown.
    """
    def __init__(self, parameters):
        self.parameters = parameters

        self.states = []  # Vertices. List of states (those can be of any type, pixels for instance).
        self.transitions = []  # Edges. Contains couples: (state indices i.e. integers, values of transitions).

        self.max_degree = 0  # maximum number of degrees (degree = number of edges that leave the node).
        self.current_state_index = None  # index of the current state
        self.number_explorations = []  # number of explorations done for each state

        # the path that the agent is currently following
        self.current_path = []

    def __str__(self):
        """
        :return: a string containing a representation of the graph
        """
        s = "\n"
        for idx in range(len(self.states)):
            if idx == self.current_state_index:
                s += red + str(idx) + white

            else:
                s += str(idx)

            s += ": " + str(self.transitions[idx]) + "\n"
        return s[:-1]

    def reset(self, new_state):
        """
        add the new_state in the archive if it does not exist and update the current_state_index.
        *NEVER* make a new transition when resetting.
        :param new_state:
        :return:
        """
        if not self.states:  # first state added
            self.states = [new_state]
            self.transitions = [[]]
            self.number_explorations = [0]
            self.current_state_index = 0

        else:
            new_state_index = find_element_in_list(new_state, self.states)  # bottleneck. Maybe hash states.

            if new_state_index is None:
                self.states.append(new_state)
                self.transitions.append([])
                self.number_explorations.append(0)
                new_state_index = len(self.states) - 1

            self.current_state_index = new_state_index

    def update_policy(self, new_state, value):
        edge_value = value + self.parameters["edge_cost"]
        new_state_index = find_element_in_list(new_state, self.states)  # bottleneck. Maybe hash states.

        if new_state_index is None:

            self.states.append(new_state)
            self.number_explorations.append(0)

            new_state_index = len(self.states) - 1
            self.transitions.append([])  # new edge without vertex
            self.transitions[self.current_state_index].append((new_state_index, edge_value))  # new vertex with a value

        else:
            if new_state_index not in [t[0] for t in self.transitions[self.current_state_index]]:
                self.transitions[self.current_state_index].append((new_state_index, edge_value))

        self.update_max_degree()

        self.current_state_index = new_state_index

    def update_max_degree(self):
        degree = len(self.transitions[self.current_state_index])
        if degree > self.max_degree:
            assert degree == self.max_degree + 1, "max_degree is not updated correctly"
            self.max_degree += 1

    def get_max_number_successors(self):
        return self.max_degree

    def get_current_state(self):
        """
        :return: the current data of the current node
        """
        if self.current_state_index is None:
            return None

        else:
            return self.states[self.current_state_index]

    def get_next_state(self, option_index):
        (next_state_index, _) = self.transitions[self.current_state_index][option_index]
        return self.states[next_state_index]

    def find_best_action(self, train_episode=None, verbose=False):
        verbose = False
        print(self.number_explorations)
        if self.current_state_index is None or not self.transitions[self.current_state_index]:
            return None  # explore

        if len(self.current_path) <= 1:  # The current path is empty, make a new path.

            # compute the global best path from the current state index
            predecessors, distances = self.find_best_path(self.current_state_index)
            # make a new path for exploring if needed
            state_idx_to_explore = self.get_state_to_explore(distances)

            if np.random.rand() < self.parameters["probability_random_action_manager"] and train_episode is not None \
                    and state_idx_to_explore is not None:

                self.current_path = self.make_sub_path(predecessors, self.current_state_index, state_idx_to_explore)
                # then explore this state
                self.current_path.append(None)
                self.number_explorations[state_idx_to_explore] += 1
                print(red + "explore" + white)
                if verbose:
                    print(self)
                    print("target state to explore " + str(state_idx_to_explore))
                    print("new_path " + str(self.current_path))

            else:
                # make a new path for exploiting
                # compute the vertices with the highest value, excluding the root
                distances[self.current_state_index] = -float("inf")
                most_valued_vertices = np.nonzero(np.array(distances) == np.max(distances))[0]
                # choose at *random* among the most valuable vertices
                most_valued_vertex = np.random.choice(most_valued_vertices)

                assert self.current_state_index != most_valued_vertex, \
                    "there is a transition from current_state_index but the max distance is - inf !"
                if verbose:
                    print("most_valued_vertex: " + str(most_valued_vertex))
                self.current_path = self.make_sub_path(predecessors, self.current_state_index, most_valued_vertex)
                if verbose:
                    print(self)
                    print("target = " + str(most_valued_vertex))
                    print("new_path " + str(self.current_path))

            # self.current_path has length > 1
            self.current_path.pop(0)
            next_state = self.current_path[0]
            # return index of the next option
            if next_state is None:
                return None
            else:
                if verbose:
                    print("option number : " +
                          str([t[0] for t in self.transitions[self.current_state_index]].index(next_state)))
                return [t[0] for t in self.transitions[self.current_state_index]].index(next_state)

        else:  # follow the existing path
            if verbose:
                print(self)
                print("existing path " + str(self.current_path))
                print("next target " + str(self.current_path[1]))

            a = self.current_path.pop(0)
            if self.current_state_index == a:  # we are in the right state, follow the rest of the path
                next_state = self.current_path[0]  # next state of the path
                if next_state is None:
                    return None
                else:  # return the option to go toward the next state
                    if verbose:
                        print("option number : " +
                              str([t[0] for t in self.transitions[self.current_state_index]].index(next_state)))

                    return [t[0] for t in self.transitions[self.current_state_index]].index(next_state)

            else:  # we are out of the path, make a new one and start again
                self.current_path = []
                return self.find_best_action(train_episode)

    def find_best_path(self, root):
        """
        Bellman-Ford algorithm to get the longest path (highest value)
        :param root: the origin of the path
        :return: the longest path from the root.
        """
        number_vertices = len(self.states)
        distances = [-float("inf")] * number_vertices
        distances[root] = 0
        predecessors = [None] * number_vertices

        for _ in range(number_vertices - 1):
            for origin in range(number_vertices):
                for (target, value) in self.transitions[origin]:
                    if distances[target] < distances[origin] + value:
                        distances[target] = distances[origin] + value
                        predecessors[target] = origin

        return predecessors, distances

    def make_sub_path(self, predecessors, origin, target):
        print("origin = " + str(origin) + " target = " + str(target))

        if origin == target:
            return [target]
        else:
            return self.make_sub_path(predecessors, origin, predecessors[target]) + [target]

    @staticmethod
    def _get_connected_component(distances):
        return np.nonzero(np.array(distances) != -float("inf"))[0]

    def _get_unexplored_states(self, path):
        return [state for state in path if self.number_explorations[state] < self.parameters["max_explore"]]

    def get_state_to_explore(self, distances):
        """
        return None if no state to explore available
        return the state index to explore which is a state accessible from current_state and is the least explored
        :return:
        """
        path = self._get_connected_component(distances)
        unexplored_states = self._get_unexplored_states(path)
        if unexplored_states:
            number_explorations = [self.number_explorations[explorable_state] for explorable_state in unexplored_states]
            return unexplored_states[int(np.argmin(number_explorations))]

        else:
            return None

#
# class GraphIntrinsicMotivation(GraphPlanningPolicyManager):
#     """
#     now the exploration is performed by just adding the inverse of the square root of the number of exploration in
#     the transitions values
#     """
#
#     def update_policy(self, new_state, value):
#         new_state_index = find_element_in_list(new_state, self.states)  # bottleneck. Maybe hash states.
#
#         if new_state_index is None:
#             edge_value = value + self.parameters["edge_cost"] + 1
#
#             self.states.append(new_state)
#             self.number_explorations.append(0)
#
#             new_state_index = len(self.states) - 1
#             self.transitions.append([])  # new edge without vertex
#             self.transitions[self.current_state_index].append((new_state_index, edge_value))  # new vertex with a value
#
#         else:
#             if new_state_index not in [t[0] for t in self.transitions[self.current_state_index]]:
#                 self.transitions[self.current_state_index].append((new_state_index, edge_value))
#
#         self.update_max_degree()
#
#         self.current_state_index = new_state_index
#
#     def find_best_action(self, train_episode=None, verbose=False):
#         verbose = True
#         if self.current_state_index is None or not self.transitions[self.current_state_index]:
#             return None  # explore
#
#         if len(self.current_path) <= 1:  # The current path is empty, make a new path.
#
#             # compute the global best path from the current state index
#             predecessors, distances = self.find_best_path(self.current_state_index)
#
#             if train_episode is not None and np.random.rand() < self.parameters["probability_random_action_manager"]:
#                 # make a new path for exploring
#                 state_to_explore = self.get_state_to_explore(distances)
#                 if state_to_explore is not None:
#                     self.current_path = self.make_sub_path(predecessors, self.current_state_index, state_to_explore)
#                     # then explore this state
#                     self.current_path += [None]
#                     if verbose:
#                         print(self)
#                         print("target state to explore " + str(state_to_explore))
#                         print("new_path " + str(self.current_path))
#
#             else:
#                 # make a new path for exploiting
#                 # compute the vertices with the highest value, excluding the root
#                 distances[self.current_state_index] = -float("inf")
#                 most_valued_vertices = np.nonzero(np.array(distances) == np.max(distances))[0]
#                 # choose at *random* among the most valuable vertices
#                 most_valued_vertex = np.random.choice(most_valued_vertices)
#
#                 assert self.current_state_index != most_valued_vertex, \
#                     "there is a transition from current_state_index but the max distance is - inf !"
#                 if verbose:
#                     print("most_valued_vertex: " + str(most_valued_vertex))
#                 self.current_path = self.make_sub_path(predecessors, self.current_state_index, most_valued_vertex)
#                 if verbose:
#                     print(self)
#                     print("target = " + str(most_valued_vertex))
#                     print("new_path " + str(self.current_path))
#
#             # self.current_path has length > 1
#             self.current_path.pop(0)
#             next_state = self.current_path[0]
#             # return index of the next option
#             if next_state is None:
#                 return None
#             else:
#                 if verbose:
#                     print("option number : " +
#                           str([t[0] for t in self.transitions[self.current_state_index]].index(next_state)))
#                 return [t[0] for t in self.transitions[self.current_state_index]].index(next_state)
#
#         else:  # follow the existing path
#             if verbose:
#                 print(self)
#                 print("existing path " + str(self.current_path))
#                 print("next target " + str(self.current_path[1]))
#
#             a = self.current_path.pop(0)
#             if self.current_state_index == a:  # we are in the right state, follow the rest of the path
#                 next_state = self.current_path[0]  # next state of the path
#                 if next_state is None:
#                     return None
#                 else:  # return the option to go toward the next state
#                     if verbose:
#                         print("option number : " +
#                               str([t[0] for t in self.transitions[self.current_state_index]].index(next_state)))
#
#                     return [t[0] for t in self.transitions[self.current_state_index]].index(next_state)
#
#             else:  # we are out of the path, make a new one and start again
#                 self.current_path = []
#                 return self.find_best_action(train_episode)