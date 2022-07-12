""" Python Standard Library """
import copy
from operator import itemgetter
""" Third Party Imports """
""" Local Imports """


class ODEFlowTerm:
    """Generic class to represent a flow between compartments in the compartmental model, which also corresponds to a
    term in one or more ODE's defining the model

    The different subclasses of ODEFlowTerm correspond with different types of ODE term:
    - Constant: the flow value is not proportional to the value in any compartment, but may vary over time.
    - Linear: the flow value is equal to a linear combination of values in a subset of model compartments
    - Nonlinear: the flow value is equal to a linear combination of values in a subset of model compartments, scaled by the sum of values in some other subset of compartments
    - Scaled nonlinear: like nonlinear, but we take a WEIGHTED sum of the scaling compartments

    """
    def __init__(self, from_cmpt_idx, to_cmpt_idx):
        self.from_cmpt_idx = from_cmpt_idx
        self.to_cmpt_idx = to_cmpt_idx

    def flow_val(self, t_int, y):
        """Get the value of the flow for a given t and model state

        Args:
            t_int: time at which you want the flow
            y: current state vector
        """
        pass

    def add_to_constant_vector(self, vector, t_int):
        """Update the provided vector of constants to include this flow term

        Args:
            vector: vector to update
            t_int: time at which to update the vector
        """
        pass

    def add_to_linear_matrix(self, matrix, t_int):
        """Update the provided matrix representing linear flows to include this flow term

        Args:
            matrix: matrix to update
            t_int: time at which to update the matrix
        """
        pass

    def add_to_nonlinear_matrices(self, matrices: list, t_int):
        """Update the provided dictionary of matrices representing nonlinear flows to include this flow term

        matrices is a dictionary whos keys are a tuple representing all scaling compartments, and values are the matrices which give the linear combination of existing compartments
        Each set of scaling compartments (making up the key) will be multiplied by the linear combination of compartments stored in the dictionary's value

        Args:
            matrices: dictionary of matrices to update
            t_int: time at which to update the matrices
        """
        pass

    def deepcopy(self):
        """perform a deep copy of this flow term

        Returns: an identical but different flow term to this one

        """
        return self.build(**{k: copy.deepcopy(v) for k, v in self.__dict__.items()})

    @classmethod
    def build(cls, from_cmpt_idx, to_cmpt_idx, coef_by_t=None, scale_by_cmpts_idxs=None,
              scale_by_cmpts_coef_by_t=None, constant_by_t=None):
        """Create the appropriate subclass, based on whether the appropriate coefficients and scaling compartments are specified

        Args:
            from_cmpt_idx: list of indices for the compartments this flow is coming FROM
            to_cmpt_idx: list of indices for the compartments this flow is going TO
            coef_by_t: a dictionary whose keys are time and whose values are a flow multiplier to be applied at that time
            scale_by_cmpts_idxs: list of indices for the scaling compartments which sum will multiply the flow
            scale_by_cmpts_coef_by_t: list of dictionaries. each gives weights (values) for each time (keys) for one of the scaling compartments; allows nonuniform weighting of the scaling compartments
            constant_by_t: dictionary with time as keys and a flow as values, used to define a constant flow term

        Returns: An instance of one of the ODEFlowTerm subclasses, chosen appropriately based on which arguments were specified

        """
        if constant_by_t is not None:
            return ConstantODEFlowTerm(from_cmpt_idx, to_cmpt_idx, constant_by_t=constant_by_t)
        elif scale_by_cmpts_coef_by_t is not None:
            return WeightedScaledODEFlowTerm(from_cmpt_idx, to_cmpt_idx, coef_by_t=coef_by_t,
                                             scale_by_cmpts_idxs=scale_by_cmpts_idxs,
                                             scale_by_cmpts_coef_by_t=scale_by_cmpts_coef_by_t)
        elif scale_by_cmpts_idxs is not None:
            return ScaledODEFlowTerm(from_cmpt_idx, to_cmpt_idx, coef_by_t=coef_by_t,
                                     scale_by_cmpts_idxs=scale_by_cmpts_idxs)
        else:
            return LinearODEFlowTerm(from_cmpt_idx, to_cmpt_idx, coef_by_t=coef_by_t)


class ConstantODEFlowTerm(ODEFlowTerm):
    """A subclass of ODEFlowTerm for when the flow is just a number and doesn't depend on the value in any compartment

    This flow may still change over time.

    """
    def __init__(self, from_cmpt_idx, to_cmpt_idx, constant_by_t):
        ODEFlowTerm.__init__(self, from_cmpt_idx=from_cmpt_idx, to_cmpt_idx=to_cmpt_idx)
        self.constant_by_t = constant_by_t

    def flow_val(self, t_int, y):
        """Get the value of the flow for a given t and model state

        Args:
            t_int: time at which you want the flow
            y: current state vector

        Returns: array of flows into/out of every compartment defined by this flow at time t_int

        """
        return self.constant_by_t[t]

    def add_to_constant_vector(self, vector, t_int):
        """Update the provided vector of constants to include this flow term

        Args:
            vector: vector to update
            t_int: time at which to update the vector
        """
        vector[self.from_cmpt_idx] -= self.constant_by_t[t_int]
        vector[self.to_cmpt_idx] += self.constant_by_t[t_int]


class LinearODEFlowTerm(ODEFlowTerm):
    """A subclass of ODEFlowTerm for when the flow is given by a linear combination of values in a subset of compartments

    """
    def __init__(self, from_cmpt_idx, to_cmpt_idx, coef_by_t):
        ODEFlowTerm.__init__(self, from_cmpt_idx=from_cmpt_idx, to_cmpt_idx=to_cmpt_idx)
        self.coef_by_t = coef_by_t

    def flow_val(self, t_int, y):
        """Get the value of the flow for a given t and model state

        Args:
            t_int: time at which you want the flow
            y: current state vector

        Returns: array of flows into/out of every compartment defined by this flow at time t_int

        """
        return y[self.from_cmpt_idx] * self.coef_by_t[t_int]

    def add_to_linear_matrix(self, matrix, t_int):
        """Update the provided matrix representing linear flows to include this flow term

        Args:
            matrix: matrix to update
            t_int: time at which to update the matrix
        """
        if self.coef_by_t[t_int] != 0:
            matrix[self.from_cmpt_idx, self.from_cmpt_idx] -= self.coef_by_t[t_int]
            matrix[self.to_cmpt_idx, self.from_cmpt_idx] += self.coef_by_t[t_int]


class ScaledODEFlowTerm(ODEFlowTerm):
    """A subclass of ODEFlowTerm for when the flow is a linear combination of values in a subset of compartments, multiplied by the sum of values in a set of scaling compartments

    """
    def __init__(self, from_cmpt_idx, to_cmpt_idx, coef_by_t, scale_by_cmpts_idxs):
        ODEFlowTerm.__init__(self, from_cmpt_idx=from_cmpt_idx, to_cmpt_idx=to_cmpt_idx)
        self.coef_by_t = coef_by_t
        self.scale_by_cmpts_idxs = sorted(scale_by_cmpts_idxs)

    def flow_val(self, t_int, y):
        """Get the value of the flow for a given t and model state

        Args:
            t_int: time at which you want the flow
            y: current state vector

        Returns: array of flows into/out of every compartment defined by this flow at time t_int

        """
        return y[self.from_cmpt_idx] * self.coef_by_t[t_int] * sum(itemgetter(*self.scale_by_cmpts_idxs)(y))

    def add_to_nonlinear_matrices(self, matrices, t_int):
        """Update the provided dictionary of matrices representing nonlinear flows to include this flow term

        matrices is a dictionary whos keys are a tuple representing all scaling compartments, and values are the matrices which give the linear combination of existing compartments
        Each set of scaling compartments (making up the key) will be multiplied by the linear combination of compartments stored in the dictionary's value

        Args:
            matrices: dictionary of matrices to update
            t_int: time at which to update the matrices
        """
        if self.coef_by_t[t_int] != 0:
            matrices[tuple(self.scale_by_cmpts_idxs)][self.from_cmpt_idx, self.from_cmpt_idx] -= self.coef_by_t[t_int]
            matrices[tuple(self.scale_by_cmpts_idxs)][self.to_cmpt_idx, self.from_cmpt_idx] += self.coef_by_t[t_int]


class WeightedScaledODEFlowTerm(ODEFlowTerm):
    """A subclass of ODEFlowTerm for when the flow is a linear combination of values in a subset of compartments, multiplied by a weighted sum of values in a set of scaling compartments

    """
    def __init__(self, from_cmpt_idx, to_cmpt_idx, coef_by_t, scale_by_cmpts_idxs, scale_by_cmpts_coef_by_t=None):
        ODEFlowTerm.__init__(self, from_cmpt_idx=from_cmpt_idx, to_cmpt_idx=to_cmpt_idx)
        self.coef_by_t = coef_by_t
        self.scale_by_cmpts_idxs = scale_by_cmpts_idxs
        self.scale_by_cmpts_coef_by_t = scale_by_cmpts_coef_by_t

    def flow_val(self, t_int, y):
        """Get the value of the flow for a given t and model state

        Args:
            t_int: time at which you want the flow
            y: current state vector

        Returns: array of flows into/out of every compartment defined by this flow at time t_int

        """
        return y[self.from_cmpt_idx] * self.coef_by_t[t_int] * sum(
            a * b for a, b in zip(itemgetter(*self.scale_by_cmpts_idxs)(y), self.scale_by_cmpts_coef_by_t[t_int]))

    def add_to_nonlinear_matrices(self, matrices, t_int):
        """Update the provided dictionary of matrices representing nonlinear flows to include this flow term

        matrices is a dictionary whos keys are a tuple representing all scaling compartments, and values are the matrices which give the linear combination of existing compartments
        Each set of scaling compartments (making up the key) will be multiplied by the linear combination of compartments stored in the dictionary's value, and the appropriate scaling weight

        Args:
            matrices: dictionary of matrices to update
            t_int: time at which to update the matrices
        """
        scale_by_cmpts_coef_by_t = self.scale_by_cmpts_coef_by_t[t_int] if self.scale_by_cmpts_coef_by_t else [1] * len(self.scale_by_cmpts_idxs)
        for scale_by_cmpt_idx, scale_by_cmpt_coef in zip(self.scale_by_cmpts_idxs, scale_by_cmpts_coef_by_t):
            matrices[scale_by_cmpt_idx][self.from_cmpt_idx, self.from_cmpt_idx] -= self.coef_by_t[t_int] * scale_by_cmpt_coef
            matrices[scale_by_cmpt_idx][self.to_cmpt_idx, self.from_cmpt_idx] += self.coef_by_t[t_int] * scale_by_cmpt_coef
