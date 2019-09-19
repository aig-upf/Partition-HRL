from collections import defaultdict
from abstract.utils.miscellaneous import *
from abstract.utils.sample import sample_pmf


class Node(object):
    """
    Tests: OK
    Reviewed : No
    This class creates a Node.
    A Node is connected to its parent node via the variable self.parent.
    A Node has children, stored in the list self.children.
    A node has a value and a variable data which is used to store a state representation.
    """
    def __init__(self, data, parent=None):
        input("GraphNode is a DEPRECATED CLASS")
        self.value = 0
        self.data = data  # a.k.a state

        self.parent = parent
        if self.parent is not None:
            self.parent.children.append(self)
            self.depth = self.parent.depth + 1

        else:
            self.depth = 0

        self.children = list()

    def __eq__(self, other):
        return obs_equal(self.data, other.data)

    def __repr__(self):
        return "data: " + str(self.data)

    def __str__(self):
        s = "\n" + "node " + str(hash(self.data)) + " at depth " + str(self.depth) + "\n"
        s += "value: " + str(self.value) + "\n"
        if self.parent is not None:
            s += "parent: " + str(hash(self.parent.data)) + "\n"

        else:
            s += "no parent\n"

        if self.children != list():
            s += "children: " + "["
            for child in self.children:
                s += str(hash(child.data)) + ", "

            s = s[:-2]
            s += "]" + "\n"
            for k in range(len(self.children)):
                s += "action: " + str(k) + \
                     " with value: " + str(self.children[k].value) \
                     + "\n"

        else:
            s += "no child\n"

        return s

    def depth_first(self):
        yield self
        for child in self.children:
            for node in child.depth_first():
                yield node

    def is_root(self):
        return self.parent is None

    def is_leaf(self):
        """
        :return: True if and only if self.children == [].
        """
        return not self.children

    def breadth_first(self):
        current_nodes = [self]
        while current_nodes:
            children = []
            for node in current_nodes:
                yield node
                children.extend(node.children)
            current_nodes = children

    def get_children_values(self):
        return [child.value for child in self.children]


class Tree:
    """
    Tests: OK
    Although a Node is a tree by itself, this class provides more iterators and
    quick access to the different depths of the tree, and keeps track of the root node

    characteristics of the tree are: max_width, nodes and depth
    """
    def __init__(self, root_data):
        self.root = Node(root_data)
        self.current_node = self.root

        # set the novelty_table
        self.shape = root_data.shape[:2]
        self.novelty_table = np.empty(self.shape, dtype=set)
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                self.novelty_table[i, j] = {tuple(root_data[i, j])}

        # characteristics of the tree
        self.max_width = 1
        self.nodes = list()
        self.depth = defaultdict(list)

    def __eq__(self, other_tree):
        iter_self = self.root.depth_first()
        iter_other = other_tree.root.depth_first()
        for n_self, n_other in zip(iter_self, iter_other):
            if n_other != n_self:
                return False

        # check if one of those iterators is not empty
        try:
            next(iter_self)
            return False
        except StopIteration:
            try:
                next(iter_other)
                return False
            except StopIteration:
                return True

    def __len__(self):
        return len(self.nodes)

    def __str__(self):
        s = str()
        s += "\n"
        for node in self.root.depth_first():
            if node == self.current_node:
                s += red

            else:
                s += green

            s += "".join([tab] * node.depth + ["|", str(id(node.data)) +
                                               ". depth : " + str(node.depth) +
                                               " value : " + str(node.value), '\n'])
        return s + white

    def tree_to_string(self, next_node):
        """
        transforms the tree into a string. The difference with __str__ is that we can give a different color to
        a particular node (next_node)
        :return: a string representing the tree
        """

        s = str()
        s += "\n"
        for node in self.root.depth_first():
            if node == self.current_node:
                s += red

            elif node == next_node:
                s += yellow

            else:
                s += green

            s += "".join([tab] * node.depth + ["|", str(node.data) + ". depth : " + str(node.depth), '\n'])
        return s + white

    def reset(self):
        self.current_node = self.root

    def get_max_width(self):
        return self.max_width

    def get_current_state(self):
        return self.current_node.data

    def move_to_child_node_from_index(self, index):
        self.current_node = self.current_node.children[index]

    def move_if_node_with_state(self, state):
        """
        update self.current_state if there exists a node with state
        :param state:
        :return True iff node with node.data == state is found
        """
        # to improve performances: first check the children
        for node in self.current_node.children:
            if obs_equal(node.data, state):
                self.current_node = node
                return True

        # then check all nodes
        for node in self.root.breadth_first():
            if obs_equal(node.data, state):
                self.current_node = node
                return True

        # if not found, return False
        return False

    def add_node(self, data):
        """
        Add the tree under the current_node if it is not novel (IW). Then, updates the tree characteristics.
        :param data: the data contained in the node that we have to add to the tree.
        :return the novelty
        """
        novel = self.update_novelty_table(data)
        if novel:
            node = Node(data, self.current_node)
            self._update_characteristics(node)
            self.current_node = node

        return novel

    def _update_characteristics(self, node: Node):
        """
        updates the depth, the nodes list, max_width and the current node
        :param node:
        """
        self.depth[node.depth].append(node)
        self.nodes.append(node)

        # update max_width
        if node.parent is not None and len(node.parent.children) > self.max_width:
            self.max_width += 1

    def update_novelty_table(self, state):
        """
        todo make tests
        updates the novelty table by including the elements of state in the novelty table if needed.
        :param state:
        :return: True iff the state is novel
        """
        novel = False
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):

                t = tuple(state[i, j])
                if t not in self.novelty_table[i, j]:
                    novel = True
                    self.novelty_table[i, j].add(t)

        return novel

    def get_children_values(self):
        return self.current_node.get_children_values()

    @staticmethod
    def _get_leaves(node: Node):
        """
        :param: node: if root : return all the leaves.
        else: get all the parents for each leaf and compare them to node.
        :return the leaves of the given node input.
        """
        leaves = []
        iterator = node.depth_first()

        # remove the first element if it is the root
        if node.is_root():
            next(iterator)

        for child in iterator:
            if child.is_leaf():
                leaves.append(child)

        return leaves

    def _get_child_index_to_leaf(self, leaf: Node):
        """
        This function gets a child index of current_node.
        This child is a parent of leaf.
        :param leaf:
        :return: an integer between 0 and len(self.current_node.children) - 1
        """
        while leaf.parent != self.current_node:
            leaf = leaf.parent

        return leaf.parent.children.index(leaf)

    def _get_probability_leaves(self):
        """
        todo : possible to put another distribution
        for all leaves from current_node, computes the probability of selecting a leaf.
        These probabilities are proportional to the depth of the leaf (computed from input node)
        :return: the probabilities of selecting a leaf, all the leaves of the tree which has input node as a parent
        """
        assert not(self.current_node.is_leaf())

        leaves = Tree._get_leaves(self.current_node)
        probability_leaves = np.zeros(len(leaves))
        idx = -1
        for leaf in leaves:
            idx += 1
            probability_leaves[idx] = (leaf.depth - self.current_node.depth)

        probability_leaves /= np.sum(probability_leaves)

        return probability_leaves, leaves

    def get_random_child_index(self):
        """
        gives the index of a child, selected according to its probability, computed with _get_probability_leaves
        :return: an index from list self.children
        """
        probability_leaves, leaves = self._get_probability_leaves()
        selected_leaf = leaves[sample_pmf(probability_leaves)]
        return self._get_child_index_to_leaf(selected_leaf)

    def get_child_data_from_index(self, child_index):
        return self.current_node.children[child_index].data
