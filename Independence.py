
import numpy as np


# ---------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------

# Helper functions

def get_non_crossing_partitions(n: int):
    '''
    Devuelve un generador con las Non-Crossing Partitions de n elementos.

    The Vi are the blocks of π ∈ P(n). π is non-crossing if we do not have
    p1 < q1 < p2 < q2
    such that p1, p2 are in a same block,
    q1, q2 are in a same block, but those two blocks are different.

    By NC(n) we will denote the set of all non-crossing partitions of [n].

    Basado en:
    https://stackoverflow.com/questions/75573540/generating-noncrossing-partitions-in-python
    '''
    # hacer una lista con numeros {1, 2, ..., n}
    elements = list(range(1, n+ 1))
    # obtener NC(n)
    # ejecutar la funcion auxiliar con la lista [n, n-1, ..., 1]
    # para ir sacando los elementos desde 1 hasta 1
    yield from _make_partitions(sorted(elements, reverse=True), [], [])


def _make_partitions(elements: list[int],
                     active_blocks: list[list[int]],
                     inactive_blocks: list[list[int]]):
    '''
    Funcion Auxiliar para get_non_crossing_partitions.

    elements: lista de elementos que falta de asignar a un bloque
    active_blocks: bloques a los que se pueden asignar mas elementos
    inactive_blocks: bloques a los que no se pueden asignar elementos (crearia crossing).

    Devuelve un generador con todas las particiones que se pueden obtener
    de asignar los elementos a algun bloque activo, y que no generan crossing.
    '''

    # 0) Caso base: Ya no hay elementos por asignar
    # la particion son los bloques activos y los no activos
    if not elements:
        yield active_blocks + inactive_blocks
        return

    # Tomar el elemento mas pequeño que falta por asignar
    elem = elements.pop()
    # para asignar este elemento hay dos posibilidades
    # 1) Iniciar un nuevo bloque con solo este elemento
    # 2) Meter este elemento a un bloque activo

    # 1) Iniciar un nuevo bloque con solo este elemento
    # hacer un bloque nuevo, ponerlo activo
    active_blocks.append([elem])
    # devolver todas las particiones con los elementos y bloques activos actualizados
    yield from _make_partitions(elements, active_blocks, inactive_blocks)
    # volver a tener los bloques activos originales, quitar el [elem]
    active_blocks.pop()

    # 2) Meter este elemento a un bloque activo

    # ver cuantos bloques activos se tienen
    size = len(active_blocks)

    # para cada bloque activo
    # (ordenado del que tiene menores elementos a mayores elementos)
    for part in active_blocks[::-1]:
        # meter el elemento a este bloque
        part.append(elem)
        # obtener todas las particiones que se siguen de esto
        yield from _make_partitions(elements, active_blocks, inactive_blocks)
        # quitar el elemento de este bloque
        part.pop()

        # ahora se va a explorar meter el elemento a otros bloques activos
        # si se mete a cualquier bloque activo de los que faltan en el for
        # entonces el bloque al que se mete tiene elementos menores que los de este bloque
        # entonces, despues de eso ya no se podria meter nada a este bloque
        # pues generaria un crossing
        # entonces marcarlo como inactivo
        # es decir: meter el bloque actual (el del for) en los inactivos, ya no activo
        inactive_blocks.append(active_blocks.pop())

    # en el for pasado se pasaron varios bloques activos a los inactivos
    # revertir esto
    for _ in range(size):
        active_blocks.append(inactive_blocks.pop())

    # volver a meter el elemento que se habia sacado al principio, para no afectar
    elements.append(elem)


def get_mixed_parameters(x, y, n: int) -> list[tuple]:
    """
    Computes tuples of length at most n with a mix of x and y
    n >= 2

    Each tuple should contain at lest one x and at least one y.
    """

    # store tuples according to their length
    # start with the case n = 1
    mixed_parameters_dict: dict[int, list[tuple]] = {1: [(x,), (y,)]}

    # consider length of at most n
    for p in range(2, n + 1):
        # compute tuples of mixed parameters with length p
        tuples_length_p: list[tuple] = []

        # for each tuple of length p-1
        for tuples_length_p_minus_1 in mixed_parameters_dict[p - 1]:
            # add x and y to the list
            tuples_length_p.append(tuples_length_p_minus_1 + (x,))
            tuples_length_p.append(tuples_length_p_minus_1 + (y,))

        # the tuples of length p have been computed
        mixed_parameters_dict[p] = tuples_length_p

    # filter only the tuples that contain both variables
    result: list[tuple] = []
    for p, tuples_length_p in mixed_parameters_dict.items():
        for tuple_mixed_parameters in tuples_length_p:
            if x in tuple_mixed_parameters and y in tuple_mixed_parameters:
                result.append(tuple_mixed_parameters)

    return result


# ---------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------


class Free_Independence:
    '''
    Evaluate if two variables A, B are
    freely independent by calculating mixed cumulants
    '''

    def __init__(self, variable_A, variable_B, functional_phi, product):
        """
        Working in a (A,ϕ) non-commutative probability space

        Variables A and B are elements of the algebra
        product: It is a function that acts as a product in the algebra
        functional_phi: It is a tracial linear functional from the algebra
                        to the complex numbers
        """

        # add the variables to the atributes
        self.A = variable_A
        self.B = variable_B
        self.variables = {"A": self.A, "B": self.B}

        # store all the evaluations of cumulants
        # for each possible combination of parameters
        # keys = parameters for the cumulant
        # value = evaluation of the cumulant on those parameters
        self.cumulant_evaluation: dict[tuple, float] = dict()

        # store the functional phi
        self.phi = functional_phi

        # store the evaluations
        # of diferent arguments (multiplication of variables)
        # keys = variables that has to be multiplied and evaluated at phi
        # value = evaluation of phi on that multiplication
        self.phi_evaluation: dict[tuple, float] = dict()

        # store the multplication
        self.product = product

        # in order to check how useful it is to store cumulant evaluations
        self.saved_cumulant_evaluation = 0
        self.not_saved_cumulant_evaluation = 0

    def check_mixed_cumulants(self, max_order: int,
                              print_results: bool = True) -> dict[tuple, float]:
        """
        Compute the mixed free cumulants up to a specific order
        """

        # store the results for all mixed cumulants
        results_mixed_cumulants: dict[tuple, float] = dict()

        # take all the arguments for mixed cumulants
        arguments_cumulants: list[tuple] = get_mixed_parameters("A", "B", max_order)

        # for each mixed cumulant
        for argument in arguments_cumulants:

            # if a cyclic permutation has been evaluated, skip
            if self.search_key_with_permutations(results_mixed_cumulants,
                                                 argument) is not None:
                continue

            # a cyclic permutation has not been evaluated yet

            # evaluate
            res_mixed_cumulant = self.evaluate_cumulant(argument)
            # save and print
            results_mixed_cumulants[argument] = res_mixed_cumulant
            if print_results:
                print(f"κ({argument}) = {res_mixed_cumulant}")

        # return the evaluation for each mixed cumulant
        return results_mixed_cumulants

    def search_key_with_permutations(self, dictionary: dict[tuple, float],
                                     search_key: tuple):
        """
        Checks if the given `search_key` or any of its cyclic permutations
        exists as a key in the provided dictionary.

        Parameters:
        - dictionary (dict[tuple, float]): A dictionary with tuples as keys
          and floats as values.
        - search_key (tuple): A tuple to search for as a key or as a cyclic permutation
          of an existing key.

        Returns:
        - float: The value associated with the matching key (if found).
        - None: If neither the `search_key` nor its cyclic permutations are found.
        """

        # length of search key
        n = len(search_key)

        # generate cyclic permutations of the search key (there are n)
        for i in range(n):

            # create a ciclyc permutation of the search key
            cyclic_permutation = search_key[i:] + search_key[:i]

            # check if this permutation is part of the dict
            if cyclic_permutation in dictionary:
                return dictionary[cyclic_permutation]

        # if all the cyclic permutations are missing from the dictionary
        return None

    def evaluate_cumulant(self, arg: tuple) -> float:
        """
        Given an argument (tuple of length n containing "A" or "B")
        Evaluate the respective cumulant k_n(arg)
        """

        # if the value for that cumulant has already been computed
        existing_value = self.search_key_with_permutations(dictionary=self.cumulant_evaluation,
                                                           search_key=arg)
        if existing_value is not None:
            # just return it
            self.saved_cumulant_evaluation += 1
            return existing_value

        # if not, then evaluate it
        self.not_saved_cumulant_evaluation += 1
        # (despejando la formula 2.9 de Free probability and Random Matrices pagina 40)

        # evaluate phi in the multiplication of the arguments
        phi_eval = self.evaluate_phi(arg)
        cumulant_result = phi_eval

        # now evaluate the cumulant for each non crossing partition
        n = len(arg)
        for partition in get_non_crossing_partitions(n):

            # do not consider the partition with all elements in one block
            # that gives exactly the cumulant evaluation that we are calculating
            if len(partition) == 1:
                continue

            # evaluate the cumulant using that partition, substract the corresponding value to the result
            cumulant_result -= self.evaluate_cumulant_partition(arg, partition)

        # after iterating in all NC(n) the result is ready
        # save it and return
        self.cumulant_evaluation[arg] = cumulant_result
        return cumulant_result

    def evaluate_cumulant_partition(self, argument: tuple, partition: list) -> float:
        """
        Given an argument (tuple of length n containing "A" or "B")
        and a non-crossing partition of the elements in [n]
        Evaluate the respective cumulant k_n(arg) with respect to the partition
        """

        # multiply for each block of the partition
        partition_cumulant_result = 1

        for block in partition:
            # take the corresponding variables
            block_variables = tuple(argument[idx - 1] for idx in block)  # -1 for 0 index in arg
            # evaluate the cumulant on those variables and multiply
            partition_cumulant_result *= self.evaluate_cumulant(block_variables)

        return partition_cumulant_result

    def evaluate_phi(self, argument_tuple: tuple) -> float:
        """
        Given an argument (tuple of length n containing "A" or "B")
        Evaluate phi on the multiplication of those arguments
        """

        # if the value for that phi evaluation has already been computed
        existing_value = self.search_key_with_permutations(dictionary=self.phi_evaluation,
                                                           search_key=argument_tuple)
        if existing_value is not None:
            # just return it
            return existing_value

        # if not, then evaluate it

        # transform the tuple argument to the real argument (multiplication)
        argument = self.multiply_argument_tuple(argument_tuple)
        # evaluate
        phi_result = self.phi(argument)

        # save and return
        self.phi_evaluation[argument_tuple] = phi_result
        return phi_result

    def multiply_argument_tuple(self, variable_tuple: tuple):
        """
        Given a tuple with "A" or "B"
        multiply all variables, on that respective order
        """

        # if there is only one argument
        if len(variable_tuple) == 1:
            # just return that variable
            return self.variables[variable_tuple[0]]

        # if there is more than one element
        # multiply them all
        multiplication = self.variables[variable_tuple[0]]
        for var in variable_tuple[1:]:
            multiplication = self.product(multiplication, self.variables[var])

        # return the multiplication
        return multiplication


# ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------

class OV_Independence:
    '''
    Evaluate if two variables A, B are
    freely independent with respect a conditional expectation
    by calculating mixed cumulants
    '''

    def __init__(self, variable_A: dict[tuple, any], variable_B: dict[tuple, any],
                 d: int, functional_phi, product):
        """
        Working in a (A,E,B) operator-valued probability space

        A = Md(C)
        B = Md(Complex) ⊂ Md(C)
        E = id ⊗ ϕ : Md(C) → Md(Complex) is a conditional expectation

        A is composed of dxd matrices of elements of C
        C is an algebra with funcional ϕ
        ((C,ϕ) is a non-commutative probability space)
        Therefore, the elements in the algebra of interest (A)
        are just matrices of elements in C

        Variables A and B are elements of the algebra A
            therefore they are matrices dxd of elements in C
            they are represented as dictionaries
            where the coordinates of an entry is the key  (e.g (1, 1) is a key, that gives the first entry)
            and the element in C is the corresponding value
        d: dimension of the matrices in the algebra A
        product: It is a function that acts as a product in the algebra C
        functional_phi: It is the tracial linear functional C → Complex
        """

        # add the variables to the atributes
        self.A = variable_A
        self.B = variable_B
        self.variables = {"A": self.A, "B": self.B}

        # store all the evaluations of operated-value cumulants
        # for each possible combination of parameters
        # keys = parameters for the cumulant (combination of A and B)
        # value = evaluation of the cumulant on those parameters (matrices)
        self.ov_cumulant_evaluation: dict[tuple, any] = dict()

        # store all the evaluations of cumulants
        # for each possible combination of parameters
        # keys = parameters for the cumulant (combination of blocks of A or B)
        # value = evaluation of the cumulant on those parameters (complex numbers)
        self.cumulant_evaluation: dict[tuple, float] = dict()

        # store the functional phi
        self.phi = functional_phi

        # store d
        self.d = d

        # store the evaluations
        # of different arguments of C (multiplication of variables)
        # as the arguments are elemtens of C, there are multiplication of blocks of A or B
        # keys = variables that has to be multiplied and evaluated at phi
        # value = evaluation of phi on that multiplication
        self.phi_evaluation: dict[tuple, float] = dict()

        # store the multplication in C
        self.product = product

        # in order to check how useful it is to store operated-valie cumulant evaluations
        self.ov_saved_cumulant_evaluation = 0
        self.ov_not_saved_cumulant_evaluation = 0
        # in order to check how useful it is to store cumulant evaluations
        self.saved_cumulant_evaluation = 0
        self.not_saved_cumulant_evaluation = 0

    def check_mixed_cumulants(self, max_order: int,
                              print_results: bool = True) -> dict[tuple, any]:
        """
        Compute the operated-value mixed free cumulants of A and B up to a specific order
        """

        # store the results for all operated-value mixed cumulants
        results_ov_mixed_cumulants: dict[tuple, any] = dict()

        # take all the arguments for mixed cumulants
        arguments_ov_cumulants: list[tuple] = get_mixed_parameters("A", "B", max_order)

        # for each operated-value mixed cumulant
        for argument in arguments_ov_cumulants:

            # compute it
            res_ov_mixed_cumulant = self.evaluate_ov_cumulant(argument)

            # store on the dictionary and print
            results_ov_mixed_cumulants[argument] = res_ov_mixed_cumulant
            if print_results:
                print(f"κ^(B)({argument}) = ")
                print(res_ov_mixed_cumulant)
                print("--")

        # return the evaluation for each mixed cumulant
        return results_ov_mixed_cumulants

    def evaluate_ov_cumulant(self, arg: tuple) -> any:
        """
        Evaluate an operated-value cumulant in a specific argument.
        The argument is a tuple containing A or B
        """

        # if the operated-value cumulant has already been computed
        if arg in self.ov_cumulant_evaluation.keys():
            # just return it
            self.ov_saved_cumulant_evaluation += 1
            return self.ov_cumulant_evaluation[arg]

        # if not, then evaluate it
        self.ov_not_saved_cumulant_evaluation += 1

        # start with an empty matrix
        res_ov_cumulant = np.zeros((self.d, self.d))

        # compute every entry
        for i in range(self.d):
            for j in range(self.d):
                # compute the value for this entry
                res_ov_cumulant[i, j] = self.evaluate_ov_cumulant_entry(arg, i + 1, j + 1)

        # save the result in the dictionary
        self.ov_cumulant_evaluation[arg] = res_ov_cumulant
        return res_ov_cumulant

    def evaluate_ov_cumulant_entry(self, argument: tuple, i: int, j: int) -> float:
        """
        Compute the entry (i,j) of the matriz resulting
        of evaluating the operated-value cumulant on
        the arguments specified

        i, j = 1, ..., d (using index starting from 1 and endings at d)

        Equation 9.23 Free Probability and Random Matrices (pag 242)
        """

        # how many arguments there are
        n = len(argument)

        # Construct all the arguments for cumulants on the blocks (n arguments)

        # for just one argument it is just the blok of the argument
        if n == 1:
            arguments_cumulant: list[tuple] = [(f"{argument[0]}_{i}_{j}",)]
        # in general for at least two arguments
        else:
            # 1) Start with the first entry
            arguments_cumulant: list[tuple] = [(f"{argument[0]}_{i}_{i2}",)
                                               for i2 in range(1, self.d + 1)]

            # 2) Add from the second entry to the second to last
            for k in range(1, n - 1):
                # compute the arguments up to that entry
                arguments_cumulant_new: list[tuple] = [arg + (f"{argument[k]}_{arg[-1].split('_')[-1]}_{i_inter}",)
                                                       for arg in arguments_cumulant
                                                       for i_inter in range(1, self.d + 1)]

                # now these are the arguments to consider
                arguments_cumulant = arguments_cumulant_new

            # 3) Add the final entry
            arguments_cumulant_new: list[tuple] = [arg + (f"{argument[-1]}_{arg[-1].split('_')[-1]}_{j}",)
                                                   for arg in arguments_cumulant]
            # there are the final arguments to consider
            arguments_cumulant = arguments_cumulant_new

        # print(arguments_cumulant)
        # assert len(arguments_cumulant) == self.d**(n-1)

        # evaluate the cumulant for each argument and sum
        res_ov_cumulant_entry: float = 0.0
        for arg in arguments_cumulant:
            # add the respective cumulant evaluation
            res_ov_cumulant_entry += self.evaluate_cumulant(arg)

        return res_ov_cumulant_entry

    def search_key_with_permutations(self, dictionary: dict[tuple, float],
                                     search_key: tuple):
        """
        Checks if the given `search_key` or any of its cyclic permutations
        exists as a key in the provided dictionary.
        Used to evaluate the cumulants in C (of block matrices) and function phi

        Parameters:
        - dictionary (dict[tuple, float]): A dictionary with tuples as keys
          and floats as values.
        - search_key (tuple): A tuple to search for as a key or as a cyclic permutation
          of an existing key.

        Returns:
        - float: The value associated with the matching key (if found).
        - None: If neither the `search_key` nor its cyclic permutations are found.
        """

        # length of search key
        n = len(search_key)

        # generate cyclic permutations of the search key (there are n)
        for i in range(n):

            # create a ciclyc permutation of the search key
            cyclic_permutation = search_key[i:] + search_key[:i]

            # check if this permutation is part of the dict
            if cyclic_permutation in dictionary:
                return dictionary[cyclic_permutation]

        # if all the cyclic permutations are missing from the dictionary
        return None

    def evaluate_cumulant(self, arg: tuple) -> float:
        """
        Given an argument (tuple of length n containing "A_i_j" or "B_i_j")
        Evaluate the respective cumulant k_n(arg)
        """

        # if the value for that cumulant has already been computed
        existing_value = self.search_key_with_permutations(dictionary=self.cumulant_evaluation,
                                                           search_key=arg)
        if existing_value is not None:
            # just return it
            self.saved_cumulant_evaluation += 1
            return existing_value

        # if not, then evaluate it
        self.not_saved_cumulant_evaluation += 1
        # (despejando la formula 2.9 de Free probability and Random Matrices pagina 40)

        # evaluate phi in the multiplication of the arguments
        phi_eval = self.evaluate_phi(arg)
        cumulant_result = phi_eval

        # now evaluate the cumulant for each non crossing partition
        n = len(arg)
        for partition in get_non_crossing_partitions(n):

            # do not consider the partition with all elements in one block
            # that gives exactly the cumulant evaluation that we are calculating
            if len(partition) == 1:
                continue

            # evaluate the cumulant using that partition, substract the corresponding value to the result
            cumulant_result -= self.evaluate_cumulant_partition(arg, partition)

        # after iterating in all NC(n) the result is ready
        # save it and return
        self.cumulant_evaluation[arg] = cumulant_result
        return cumulant_result

    def evaluate_cumulant_partition(self, argument: tuple, partition: list) -> float:
        """
        Given an argument (tuple of length n containing "A_i_j" or "B_i_j")
        and a non-crossing partition of the elements in [n]
        Evaluate the respective cumulant k_n(arg) with respect to the partition
        """

        # multiply for each block of the partition
        partition_cumulant_result = 1

        for block in partition:
            # take the corresponding variables
            block_variables = tuple(argument[idx - 1] for idx in block)  # -1 for 0 index in arg
            # evaluate the cumulant on those variables and multiply
            partition_cumulant_result *= self.evaluate_cumulant(block_variables)

        return partition_cumulant_result

    def evaluate_phi(self, argument_tuple: tuple) -> float:
        """
        Given an argument (tuple of length n containing "A" or "B")
        Evaluate phi on the multiplication of those arguments
        """

        # if the value for that phi evaluation has already been computed
        existing_value = self.search_key_with_permutations(dictionary=self.phi_evaluation,
                                                           search_key=argument_tuple)
        if existing_value is not None:
            # just return it
            return existing_value

        # if not, then evaluate it

        # transform the tuple argument to the real argument (multiplication)
        argument = self.multiply_argument_tuple(argument_tuple)
        # evaluate
        phi_result = self.phi(argument)

        # save and return
        self.phi_evaluation[argument_tuple] = phi_result
        return phi_result

    def multiply_argument_tuple(self, variable_tuple: tuple):
        """
        Given a tuple with "A_i_j" or "B_i_j" for some i,j
        multiply all variables, on that respective order
        """

        # if there is only one argument
        if len(variable_tuple) == 1:
            # just return that variable
            return self.get_block_element(variable_tuple[0])

        # if there is more than one element
        # multiply them all
        multiplication = self.get_block_element(variable_tuple[0])
        for var in variable_tuple[1:]:
            multiplication = self.product(multiplication, self.get_block_element(var))

        # return the multiplication
        return multiplication

    def get_block_element(self, string_argument: str) -> any:
        """
        Given a string of the form D_i_j
        where D is A or B, and i, j = 1, ..., d
        Get the corresponding block element
        """

        # separate elements
        matriz, i, j = string_argument.split("_")
        # get the coordinates as a tuple
        block_coord = tuple((int(i), (int(j))))

        return self.variables[matriz][block_coord]


# ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------


