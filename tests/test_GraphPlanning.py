__all__ = ["ManagerTest"]
import unittest
from baseline.policy.graph_planning import GraphPlanningPolicyManager


class ManagerTest(unittest.TestCase):

    def setUp(self):
        self.parameters = {"max_explore": 0,
                           "probability_random_action_manager": 0,
                           "verbose": False,
                           "edge_cost": 0}
        self.graphs = []
        self.best_paths = []

        # graph star
        self.set_graph_star()

        # graph loop
        self.graphs.append(self.get_graph_loop([1, 0.1, -1, -1, 0]))
        self.best_paths.append([0, 1, 2])

    def set_graph_star(self):
        graph = GraphPlanningPolicyManager(parameters=self.parameters)
        graph.states = range(5)
        graph.number_explorations = [0]*5
        graph.transitions = [[(1, -1), (3, 1), (4, -5)], [(4, -0.1)], [(3, -1)], [(1, -0.5)], [(2, 1)]]
        graph.current_state_index = 0
        graph.current_path = []

        self.graphs.append(graph)
        self.best_paths.append([0, 3, 1, 4, 2])

    def get_graph_loop(self, values):
        graph = GraphPlanningPolicyManager(parameters=self.parameters)
        lv = len(values)
        graph.states = range(lv)
        graph.number_explorations = [0] * lv
        graph.transitions = [[(i+1, values[i])] for i in range(lv - 1)] + [[(0, values[-1])]]
        graph.current_state_index = 0
        graph.current_path = []

        return graph

    # ------------- The tests are defined here --------------
    def test_find_best_path(self):
        for graph, best_path in zip(self.graphs, self.best_paths):
            graph.set_best_path()
            self.assertListEqual(graph.current_path, best_path)

    def test_positive_loop(self):
        graph = self.get_graph_loop(values=[1] * 5)
        try:
            graph.set_best_path()
            success = False
        except AssertionError:
            success = True

        self.assertTrue(success)