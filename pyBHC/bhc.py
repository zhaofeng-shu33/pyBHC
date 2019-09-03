from __future__ import print_function, division
import itertools as it
import numpy as np
import sys

from numpy import logaddexp
import math


class bhc(object):
    """
    An instance of Bayesian hierarchical clustering CRP mixture model.
    Attributes
    ----------
    assignments : list(list(int))
        A list of lists, where each list records the clustering at
        each step by giving the index of the leftmost member of the
        cluster a leaf is traced to.
    root_node : Node
        The root node of the clustering tree.
    lml : float
        An estimate of the log marginal likelihood of the model
        under a DPMM.
    Notes
    -----
    The cost of BHC scales as O(n^2) and so becomes inpractically
    large for datasets of more than a few hundred points.
    """

    def __init__(self, data, data_model, crp_alpha=1.0,
                 verbose=False):
        """
        Init a bhc instance and perform the clustering.

        Parameters
        ----------
        data : numpy.ndarray (n, d)
            Array of data where each row is a data point and each
            column is a dimension.
        data_model : CollapsibleDistribution
            Provides the approprite ``log_marginal_likelihood``
            function for the data.
        crp_alpha : float (0, Inf)
            CRP concentration parameter.
        verbose : bool, optional
            Determibes whetrher info gets dumped to stdout.
        """
        self.data = data
        self.data_model = data_model
        self.crp_alpha = crp_alpha

        self.verbose = verbose

        # initialize the tree
        nodes = dict((i, Node(np.array([x]), data_model, crp_alpha,
                              indexes=i))
                     for i, x in enumerate(data))
        n_nodes = len(nodes)
        start_n_nodes = len(nodes)
        assignment = [i for i in range(n_nodes)]
        self.assignments = [list(assignment)]
        rks = []
        self.lmls = []
        while n_nodes > 1:
            if self.verbose:
                sys.stdout.write("\r{0:d} of {1:d} ".format(n_nodes,
                                                            start_n_nodes))
                sys.stdout.flush()

            max_rk = float('-Inf')
            merged_node = None

            # for each pair of clusters (nodes), compute the merger
            # score.
            for left_idx, right_idx in it.combinations(nodes.keys(),
                                                       2):
                tmp_node = Node.as_merge(nodes[left_idx],
                                         nodes[right_idx])

                if tmp_node.log_rk > max_rk:
                    max_rk = tmp_node.log_rk
                    merged_node = tmp_node
                    merged_right = right_idx
                    merged_left = left_idx

            rks.append(math.exp(max_rk))

            # Merge the highest-scoring pair
            del nodes[merged_right]
            nodes[merged_left] = merged_node

            for i, k in enumerate(assignment):
                if k == merged_right:
                    assignment[i] = merged_left
            self.assignments.append(list(assignment))

            n_nodes -= 1
            self.lmls.append(denom)
        self.root_node = nodes[0]
        self.assignments = np.array(self.assignments)

        # The denominator of log_rk is at the final merge is an
        # estimate of the marginal likelihood of the data under DPMM
        self.lml = self.root_node.log_ml

    def left_run(self):
        node = self.root_node
        while node.left_child is not None:
            print(node.indexes, np.mean(node.data, axis=0), node.data.shape)
            node = node.left_child
        print(node.indexes, np.mean(node.data, axis=0), node.data.shape)

    def right_run(self):
        node = self.root_node
        while node.right_child is not None:
            print(node.indexes, np.mean(node.data, axis=0), node.data.shape)
            node = node.right_child
        print(node.indexes, np.mean(node.data, axis=0), node.data.shape)

    def find_path(self, index):
        """ find_path(index)

            Finds the sequence of left and right merges needed to
            run from the root node to a particular leaf.

            Parameters
            ----------
            index : int
                The index of the leaf for which we want the path
                from the root node.
        """
        merge_path = []
        last_leftmost_index = self.assignments[-1][index]
        last_right_incluster = (self.assignments[-1]
                                == last_leftmost_index)

        for it in range(len(self.assignments)-2, -1, -1):
            new_leftmost_index = self.assignments[it][index]

            if new_leftmost_index != last_leftmost_index:
                # True if leaf is on the right hand side of a merge
                merge_path.append("right")
                last_leftmost_index = new_leftmost_index
                last_right_incluster = (self.assignments[it]
                                        == new_leftmost_index)

            else:       # Not in a right hand side of a merge

                new_right_incluster = (self.assignments[it]
                                       == last_leftmost_index)

                if (new_right_incluster != last_right_incluster).any():
                    # True if leaf is on the left hand side of a merge
                    merge_path.append("left")
                    last_right_incluster = new_right_incluster

        return merge_path

    def sample(self, size=1):

        output = np.zeros((size, self.root_node.data.shape[1]))

        for it in range(size):

            sampled = False
            node = self.root_node

            while not sampled:

                if node.log_rk is None:     # Node is a leaf
                    output[it, :] = self.data_model.conditional_sample(
                                        node.data)
                    sampled = True

                elif np.random.rand() < math.exp(node.log_rk):
                    # sample from node
                    output[it, :] = self.data_model.conditional_sample(
                                                                    node.data)
                    sampled = True

                else:   # drop to next level
                    child_ratio = (node.left_child.nk
                                   / (node.left_child.nk+node.right_child.nk))
                    if np.random.rand() >= child_ratio:
                        node = node.right_child
                    else:
                        node = node.left_child

        return output


class Node(object):
    """ A node in the hierarchical clustering.
    Attributes
    ----------
    nk : int
        Number of data points assigned to the node
    data : numpy.ndarrary (n, d)
        The data assigned to the Node. Each row is a datum.
    data_model : idsteach.CollapsibleDistribution
        The data model used to calcuate marginal likelihoods
    crp_alpha : float
        Chinese restaurant process concentration parameter
    log_dk : float
        Used in the calculation of the prior probability. Defined in
        Fig 3 of Heller & Ghahramani (2005).
    log_pi : float
        Prior probability that all associated leaves belong to one
        cluster.
    log_ml : float
        The log marginal likelihood for the tree of the node and
        its children. This is given by eqn 2 of Heller &
        Ghahrimani (2005). Note that this definition is
        recursive.  Do not define if the node is
        a leaf.
    logp : float
        The log marginal likelihood for the particular cluster
        represented by the node. Given by eqn 1 of Heller &
        Ghahramani (2005).
    log_rk : float
        The log-probability of the merge that created the node. For
        nodes that are leaves (i.e. not created by a merge) this is
        None.
    left_child : Node
        The left child of a merge. For nodes that are leaves (i.e.
        the original data points and not made by a merge) this is
        None.
    right_child : Node
        The right child of a merge. For nodes that are leaves
        (i.e. the original data points and not made by a merge)
        this is None.
    index : int
        The indexes of the leaves associated with the node in some
        indexing scheme.
    """

    def __init__(self, data, data_model, crp_alpha=1.0, log_dk=None,
                 log_pi=0.0, log_ml=None, logp=None, log_rk=None,
                 left_child=None, right_child=None, indexes=None):
        """
        Parameters
        ----------
        data : numpy.ndarray
            Array of data_model-appropriate data
        data_model : idsteach.CollapsibleDistribution
            The data model used to calcuate marginal likelihoods
        crp_alpha : float (0, Inf)
            CRP concentration parameter
        log_dk : float
            Cached probability variable. Do not define if the node is
            a leaf.
        log_pi : float
            Cached probability variable. Do not define if the node is
            a leaf.
        log_ml : float
            The log marginal likelihood for the tree of the node and
            its children. This is given by eqn 2 of Heller &
            Ghahrimani (2005). Note that this definition is
            recursive.  Do not define if the node is
            a leaf.
        logp : float
            The log marginal likelihood for the particular cluster
            represented by the node. Given by eqn 1 of Heller &
            Ghahramani (2005).
        log_rk : float
            The probability of the merged hypothesis for the node.
            Given by eqn 3 of Heller & Ghahrimani (2005). Do not
            define if the node is a leaf.
        left_child : Node, optional
            The left child of a merge. For nodes that are leaves (i.e.
            the original data points and not made by a merge) this is
            None.
        right_child : Node, optional
            The right child of a merge. For nodes that are leaves
            (i.e. the original data points and not made by a merge)
            this is None.
        index : int, optional
            The index of the node in some indexing scheme.
        """
        self.data_model = data_model
        self.data = data
        self.nk = data.shape[0]
        self.crp_alpha = crp_alpha
        self.log_pi = log_pi
        self.log_rk = log_rk

        self.left_child = left_child
        self.right_child = right_child

        if isinstance(indexes, int):
            self.indexes = [indexes]
        else:
            self.indexes = indexes

        if log_dk is None:
            self.log_dk = math.log(crp_alpha)
        else:
            self.log_dk = log_dk

        if logp is None:    # i.e. for a leaf
            self.logp = self.data_model.\
                            log_marginal_likelihood(self.data)
        else:
            self.logp = logp

        if log_ml is None:  # i.e. for a leaf
            self.log_ml = self.logp
        else:
            self.log_ml = log_ml

    @classmethod
    def as_merge(cls, node_left, node_right):
        """ Create a node from two other nodes
        Parameters
        ----------
        node_left : Node
            the Node on the left
        node_right : Node
            The Node on the right
        """
        crp_alpha = node_left.crp_alpha
        data_model = node_left.data_model
        data = np.vstack((node_left.data, node_right.data))
        indexes = node_left.indexes + node_right.indexes
        indexes.sort()

        nk = data.shape[0]
        log_dk = logaddexp(math.log(crp_alpha) + math.lgamma(nk),
                           node_left.log_dk + node_right.log_dk)
        log_pi = -math.log1p(math.exp(node_left.log_dk
                                      + node_right.log_dk
                                      - math.log(crp_alpha)
                                      - math.lgamma(nk)))

        # Calculate log_rk - the log probability of the merge

        logp = data_model.log_marginal_likelihood(data)
        numer = log_pi + logp

        neg_pi = math.log(-math.expm1(log_pi))
        log_ml = logaddexp(numer, neg_pi+node_left.log_ml + node_right.log_ml)

        log_rk = numer-log_ml

        if log_pi == 0:
            raise RuntimeError('Precision error')

        return cls(data, data_model, crp_alpha, log_dk, log_pi,
                   log_ml, logp, log_rk, node_left, node_right,
                   indexes)
