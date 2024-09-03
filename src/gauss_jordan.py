"""
This module contains the implementation of the GaussJordanSolver class, which is designed
to solve systems of linear equations using the Gauss-Jordan elimination method
"""
from src.auxiliar_functions import convert_matrix_to_fractions, get_fraction, get_vars_list

class GaussJordanSolver:
    def __init__(self, matrix):
        self.matrix = matrix
        self.partial_matrix_solutions = []
        self.operations = []
        self.__processed_rows_indexs = []
        self.__processed_cols_indexs = []
        self.__is_solved = False


    def solve_matrix(self):
        """ Solves the matrix using the Gauss-Jordan elimination method """
        if not self.matrix:
            return

        for _ in self.matrix:
            not_null_num_index = self.__get_not_null_num_index()

            if not not_null_num_index:
                continue

            row_index, col_index = not_null_num_index

            self.__normalize_row(row_index, col_index)

            self.partial_matrix_solutions.append(convert_matrix_to_fractions(self.matrix[:]))

            self.__make_colum_ceros(row_index, col_index)

        self.__is_solved = True
        self.__order_null_colums()

    def has_solution(self):
        """ Checks if the system of equations has a solution """
        if not self.__is_solved:
            self.solve_matrix()

        for row in self.matrix:
            last_col_digit = row[-1]
            if last_col_digit != 0 and self.__all_ceros(row[:-1]):
                return False

        return True

    def print_solutions(self):
        """ Prints implicit solutions of the system of equations """
        has_solution = self.has_solution()
        print(f"System {'has' if has_solution else 'does not have'} solution\n")

        if has_solution:
            implicit_solutions = self.__get_implicit_solutions_from_matrix()
            print("Implicit Equations:")
            for eq in implicit_solutions:
                print(eq)

    def print_operations(self):
        """ Prints the operations performed on the matrix during the solution process """

        print("Operations made:\n")
        for index, matrix in enumerate(self.partial_matrix_solutions):
            print(f"Operation {index}:", self.operations[index])
            self.print_matrix(matrix)

    def print_matrix(self, matrix=None):
        """
        Prints matrix in a formatted way.

        Args:
            matrix (list of list of numbers, optional): The matrix to print. If None, prints the solver's matrix.
        """
        matrix = self.matrix if matrix is None else matrix
        max_digits = int(self.__get_max_char_amount_in_array(matrix)/2)
        print("Matrix:")
        print("-" * (max_digits + 3) * len(matrix[0]))

        for row in matrix:
            print("|", end="")
            for num in row[:-1]:
                str_el = str(num)
                padding = ((max_digits - len(str_el)) // 2)
                print(" " * padding, str_el,
                      " " * (max_digits - len(str_el) - padding), end="")

            str_el = str(row[-1])
            padding = (max_digits - len(str_el)) // 2
            print("|", " " * padding, str_el,
                  " " * (max_digits - len(str_el) - padding), "|")

        print("-" * (max_digits + 3) * len(matrix[0]))

    def update_matrix_element(self, row_index, col_index, new_value):
        if 0 <= row_index < len(self.matrix) and 0 <= col_index < len(self.matrix[0]):
            self.matrix[row_index][col_index] = new_value
        else:
            raise IndexError("Row or column index out of range.")

    def __get_implicit_solutions_from_matrix(self):
        """ Extracts implicit solutions from the matrix """

        vars_list = get_vars_list(self.matrix)
        results = []
        for row in self.matrix:
            temp_eqs = []
            if self.__all_ceros(row[:-1]):
                continue
            for index, num in enumerate(row[:-1]):
                if num != 0:
                    temp_str = f"{vars_list[index]})"
                    if num != 1:
                        temp_str = f"({num}*" + temp_str
                    else:
                        temp_str = "(" + temp_str

                    temp_eqs.append(temp_str)

            results.append("+".join(temp_eqs) + f"={row[-1]}")

        return results

    def __normalize_row(self, row_index, col_index):
        """ Normalizes a row to make the pivot element equal to 1 """

        pivot_value = self.matrix[row_index][col_index]
        if pivot_value != 1:
            self.matrix[row_index] = self.__get_divided_vector(self.matrix[row_index], pivot_value)
        self.operations.append(f"F{row_index}/({get_fraction(pivot_value)})")

    def __make_colum_ceros(self, row_index, col_index):
        """ Makes all elements in a column zero except for the pivot element """

        if not (0 <= row_index < len(self.matrix) and \
                (0 <= col_index < len(self.matrix[0]))):
            return

        if self.matrix[row_index][col_index] != 1:
            return

        for index in range(len(self.matrix)):
            if index != row_index:
                actual_num = self.matrix[index][col_index]

                if actual_num != 0:
                    self.operations.append(
                        f"F{index}"
                        f"-({get_fraction(actual_num)}"
                        f"*F{row_index})".replace("-(-", "+(", 1))

                    self.matrix[index] = [num - actual_num * self.matrix[row_index][sub_index]
                                          for sub_index, num in enumerate(self.matrix[index])]

                    self.partial_matrix_solutions.append(
                        convert_matrix_to_fractions(self.matrix[:]))

    def __get_divided_vector(self, vector, divider):
        return [num/divider for num in vector]

    def __all_ceros(self, vect):
        return all(num == 0 for num in vect)

    def __get_not_null_num_index(self):
        """ Finds the first non-zero element in the matrix that has not been processed """

        for row_index, row in enumerate(self.matrix):
            if row_index in self.__processed_rows_indexs:
                continue
            for col_index, num in enumerate(row[:-1]):
                if col_index in self.__processed_cols_indexs:
                    continue
                if num != 0:
                    self.__processed_cols_indexs.append(col_index)
                    self.__processed_rows_indexs.append(row_index)
                    return [row_index, col_index]
        return []

    def __order_null_colums(self):
        """ Orders the rows of the matrix by moving all zero rows to the bottom """

        non_null_rows = [row for row in self.matrix if not self.__all_ceros(row[:-1])]
        null_rows = [row for row in self.matrix if self.__all_ceros(row[:-1])]
        self.matrix = non_null_rows + null_rows

    def __get_max_char_amount_in_array(self, array):
        max_chars = 0
        for row in array:
            str_row = ''.join(map(str, row))
            max_chars = max(max_chars, len(str_row))

        return max_chars
 