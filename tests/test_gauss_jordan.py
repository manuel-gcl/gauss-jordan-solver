"""
This module contains unit tests for the GaussJordanSolver class.
"""
from fractions import Fraction
import pytest

from src.gauss_jordan import GaussJordanSolver


class TestGaussJordanSolver:

    @pytest.fixture
    def simple_matrix(self):
        return GaussJordanSolver([])

    @pytest.fixture
    def setup_solver(self):
        def _setup_solver(matrix):
            return GaussJordanSolver(matrix)
        return _setup_solver

    @pytest.mark.parametrize("matrix, expected_output, has_solution", [
        (
            [[2, 4, 0], [1, 2, 0]],
            [
                [Fraction(1, 1), Fraction(2, 1), Fraction(0, 1)],
                [Fraction(0, 1), Fraction(0, 1), Fraction(0, 1)]
            ], True
        ),
        (
            [[1, 2, 3], [2, 4, 6]], [
                [Fraction(1, 1), Fraction(2, 1), Fraction(3, 1)],
                [Fraction(0, 1), Fraction(0, 1), Fraction(0, 1)]],
            True
        ),
        (
            [[1, -2, 1], [2, -4, 2]], [
                [Fraction(1, 1), Fraction(-2, 1), Fraction(1, 1)],
                [Fraction(0, 1), Fraction(0, 1), Fraction(0, 1)]],
            True),
        (
            [[2, 1, -1, 8], [-3, -1, 2, -11], [-2, 1, 2, -3]], [
                [Fraction(1, 1), Fraction(0, 1),
                 Fraction(0, 1), Fraction(2, 1)],
                [Fraction(0, 1), Fraction(1, 1),
                 Fraction(0, 1), Fraction(3, 1)],
                [Fraction(0, 1), Fraction(0, 1),
                 Fraction(1, 1), Fraction(-1, 1)]
            ], True
        ),
        (
            [[1, 1, 1, 1, 10], [2, 3, 4, 5, 22], [3, 5, 2, 1, 14], [4, 2, 5, 3, 20]], [
                [Fraction(1, 1), Fraction(0, 1), Fraction(
                    0, 1), Fraction(0, 1), Fraction(84, 5)],
                [Fraction(0, 1), Fraction(1, 1), Fraction(
                    0, 1), Fraction(0, 1), Fraction(-18, 5)],
                [Fraction(0, 1), Fraction(0, 1), Fraction(
                    1, 1), Fraction(0, 1), Fraction(-76, 5)],
                [Fraction(0, 1), Fraction(0, 1), Fraction(
                    0, 1), Fraction(1, 1), Fraction(12, 1)]
            ], True
        ),
        (
            [[1, 0, 2, -1, 3, 5], [0, 1, -2, 2, -3, -1],
             [2, -2, 5, -1, 1, 7], [-1, 2, -1, 4, -2, 2], [3, -3, 1, -2, 4, 3]], [
                [Fraction(1, 1), Fraction(0, 1), Fraction(0, 1), Fraction(
                    0, 1), Fraction(0, 1), Fraction(176, 97)],
                [Fraction(0, 1), Fraction(1, 1), Fraction(0, 1), Fraction(
                    0, 1), Fraction(0, 1), Fraction(129, 97)],
                [Fraction(0, 1), Fraction(0, 1), Fraction(1, 1), Fraction(
                    0, 1), Fraction(0, 1), Fraction(124, 97)],
                [Fraction(0, 1), Fraction(0, 1), Fraction(0, 1),
                 Fraction(1, 1), Fraction(0, 1), Fraction(83, 97)],
                [Fraction(0, 1), Fraction(0, 1), Fraction(0, 1),
                 Fraction(0, 1), Fraction(1, 1), Fraction(48, 97)],
            ], True
        ),
        (
            [[1, 0, 2, -1, 3, 1, 5], [0, 1, -2, 2, -3, 2, -1], [2, -2, 5, -1, 1, 3, 7],
             [-1, 2, -1, 4, -2, 0, 2], [3, -3, 1, -2, 4, -1, 3], [1, -1, 0, 1, -1, 2, 1]], [
                [Fraction(1, 1), Fraction(0, 1), Fraction(0, 1), Fraction(
                    0, 1), Fraction(0, 1), Fraction(0, 1), Fraction(239, 148)],
                [Fraction(0, 1), Fraction(1, 1), Fraction(0, 1), Fraction(
                    0, 1), Fraction(0, 1), Fraction(0, 1), Fraction(171, 148)],
                [Fraction(0, 1), Fraction(0, 1), Fraction(1, 1), Fraction(
                    0, 1), Fraction(0, 1), Fraction(0, 1), Fraction(181, 148)],
                [Fraction(0, 1), Fraction(0, 1), Fraction(0, 1), Fraction(
                    1, 1), Fraction(0, 1), Fraction(0, 1), Fraction(137, 148)],
                [Fraction(0, 1), Fraction(0, 1), Fraction(0, 1), Fraction(
                    0, 1), Fraction(1, 1), Fraction(0, 1), Fraction(87, 148)],
                [Fraction(0, 1), Fraction(0, 1), Fraction(0, 1), Fraction(
                    0, 1), Fraction(0, 1), Fraction(1, 1), Fraction(15, 148)],
            ], True
        )
    ])
    def test_matrix_with_solution(self, setup_solver, matrix, expected_output, has_solution):
        solver = setup_solver(matrix)
        solver.solve_matrix()

        assert solver.matrix == expected_output
        assert solver.has_solution() == has_solution

    @pytest.mark.parametrize("matrix, has_solution", [
            ([[1, 0, 0, 2], [0, 1, 0, 3], [0, 0, 1, 4]], True),
            ([[1, 0, 5], [0, 1, 6]], True),
            ([[1, 0, 0, 7], [0, 1, 0, 8], [0, 0, 1, 9]], True),
            ([], True)
        ])
    def test_has_unique_solutions(self, setup_solver, matrix, has_solution):
        solver = setup_solver(matrix)
        assert solver.has_solution() == has_solution

    @pytest.mark.parametrize("matrix, has_solution", [
            ([[1, 2, 3, 4], [0, 0, 0, 0], [0, 0, 0, 0]], True),
            ([[1, 2, 3], [0, 0, 0]], True),
            ([[1, 0, 2, 4], [0, 1, 3, 5], [0, 0, 0, 0]], True),
            ([[0, 0, 0, 0]], True)
        ])
    def test_has_infinite_solutions(self, setup_solver, matrix, has_solution):
        solver = setup_solver(matrix)
        assert solver.has_solution() == has_solution

    @pytest.mark.parametrize("matrix, has_solution", [
            ([[1, 2, 3, 4], [0, 0, 0, 5], [0, 0, 0, 0]], False),
            ([[1, 1, 1], [0, 0, 1]], False),
            ([[1, 2, 3, 4], [0, 1, 1, 2], [0, 0, 0, 1]], False),
            ([[0, 0, 0, 1]], False)
        ])
    def test_has_none_solution(self, setup_solver, matrix, has_solution):
        solver = setup_solver(matrix)
        assert solver.has_solution() == has_solution

    @pytest.mark.parametrize("matrix, expected_output, has_solution", [
        (
            [[1, 1, 1], [2, 2, 3]],
            [[Fraction(1), Fraction(1, 1), Fraction(1, 1)],
             [Fraction(0, 1), Fraction(0, 1), Fraction(1, 1)]
             ], False
        ),
        (
            [[1, 1, 10], [1, 1, 5]], [
                [Fraction(1, 1), Fraction(1, 1), Fraction(10, 1)],
                [Fraction(0, 1), Fraction(0, 1), Fraction(-5, 1)]
                ], False
        ),
        (
            [[1, 2, -1, 1], [-1, 1, 2, 3], [1, 5, 0, 0]], [
                 [Fraction(1, 1), Fraction(0, 1), Fraction(-5, 3), Fraction(-5, 3)],
                 [Fraction(0, 1), Fraction(1, 1), Fraction(1, 3), Fraction(4, 3)],
                 [Fraction(0, 1), Fraction(0, 1), Fraction(0, 1), Fraction(-5, 1)]
                 ], False
        ),
        (
            [[1, 1, 1, 1, 1], [2, 2, 2, 2, 2], [3, 3, 3, 3, 3], [4, 4, 4, 4, 5]], [
            [Fraction(1, 1), Fraction(1, 1), Fraction(1, 1), Fraction(1, 1), Fraction(1, 1)],
            [Fraction(0, 1),Fraction(0, 1), Fraction(0, 1), Fraction(0, 1), Fraction(0, 1)],
            [Fraction(0, 1), Fraction(0, 1), Fraction(0, 1), Fraction(0, 1), Fraction(0, 1)],
            [Fraction(0, 1), Fraction(0, 1), Fraction(0, 1), Fraction(0, 1), Fraction(1, 1)]
            ], False
        ),(
            [[2, 3, -4, 5, -6, 7, 15], [4, 6, -8, 10, -12, 14, 30], [6, 9, -12, 15, -18, 21, 45],
             [8, 12, -16, 20, -24, 5, 60], [10, 15, -20, 25, -30, 35, 0], [3, 5, -7, 9, -11, 13, -1]], [
            [Fraction(1, 1), Fraction(0, 1), Fraction(1, 1), Fraction(-2, 1), Fraction(3, 1), Fraction(0, 1), Fraction(78, 1)],
            [Fraction(0, 1),Fraction(0, 1), Fraction(0, 1),Fraction(0, 1), Fraction(0, 1),Fraction(1, 1), Fraction(0, 1)],
            [Fraction(0, 1), Fraction(1, 1), Fraction(-2, 1), Fraction(3, 1), Fraction(-4, 1), Fraction(0, 1), Fraction(-47,1)],
            [Fraction(0, 1),Fraction(0, 1), Fraction(0, 1),Fraction(0, 1), Fraction(0, 1),Fraction(0, 1), Fraction(0, 1)],
            [Fraction(0, 1),Fraction(0, 1),Fraction(0, 1),Fraction(0, 1),Fraction(0, 1),Fraction(0, 1),Fraction(0, 1)],
            [Fraction(0, 1),Fraction(0, 1),Fraction(0, 1),Fraction(0, 1),Fraction(0, 1),Fraction(0, 1),Fraction(-75, 1)]
            ], False
        )
    ])
    def test_matrix_without_solution(self, setup_solver, matrix, expected_output, has_solution):
        solver = setup_solver(matrix)
        solver.solve_matrix()

        assert solver.matrix == expected_output
        assert solver.has_solution() == has_solution

    @pytest.mark.parametrize("initial_matrix, row_index, col_index, new_value, expected_matrix", [
        ([[1, 2], [3, 4]], 0, 1, 5, [[1, 5], [3, 4]]),
        ([[1, 2, 3], [4, 5, 6]], 1, 2, 7, [[1, 2, 3], [4, 5, 7]]),
        ([[0]], 0, 0, 1, [[1]]),
        ([[1, 2], [3, 4]], 0, 0, 10, [[10, 2], [3, 4]])
    ])
    def test_update_matrix_element(self, setup_solver ,initial_matrix, row_index, col_index, new_value, expected_matrix):
        solver = setup_solver(initial_matrix)
        solver.update_matrix_element(row_index, col_index, new_value)
        assert solver.matrix == expected_matrix

    @pytest.mark.parametrize("initial_matrix, row_index, col_index, new_value", [
        ([[1, 2], [3, 4]], -1, 0, 5),
        ([[1, 2], [3, 4]], 0, -1, 5),
        ([[1, 2], [3, 4]], 2, 0, 5),
        ([[1, 2], [3, 4]], 0, 2, 5)
    ])
    def test_update_matrix_element_out_of_range(self, setup_solver, initial_matrix, row_index, col_index, new_value):
        solver = setup_solver(initial_matrix)
        with pytest.raises(IndexError):
            solver.update_matrix_element(row_index, col_index, new_value)

    @pytest.mark.parametrize("matrix, expected", [
            ([], []),
            ([[0, 0, 0]], []),
            ([[1, 0, 0]], ["(a)=0"]),
            ([[1, 1, 2]], ["(a)+(b)=2"]),
            ([[2, 1, 3]], ["(2*a)+(b)=3"]),
            ([[0, 1, 2], [1, 0, 3]], ["(b)=2", "(a)=3"]),
            ([[1, 2, 3, 4], [0, 0, 0, 0]], ["(a)+(2*b)+(3*c)=4"]),
            ([[0, 0, 0, 4]], []),
            ([[1, 2, 3, 4], [0, 1, 1, 1]], ["(a)+(2*b)+(3*c)=4", "(b)+(c)=1"]),
        ]
    )
    def test_get_implicit_solutions_from_matrix(self, setup_solver, matrix, expected):
        solver = setup_solver(matrix)
        result = solver._GaussJordanSolver__get_implicit_solutions_from_matrix()
        assert result == expected

    @pytest.mark.parametrize(
        "matrix, row_index, col_index, expected_matrix, expected_operation", [
            ([[2, 4], [1, 3]], 0, 0, [[1, 2], [1, 3]], "F0/(2)"),
            ([[1, 2], [3, 4]], 0, 0, [[1, 2], [3, 4]], "F0/(1)"),
            ([[0.5, 1.5], [2, 3]], 0, 0, [[1, 3], [2, 3]], "F0/(1/2)"),
            ([[0, 1], [1, 2]], 0, 1, [[0, 1], [1, 2]], "F0/(1)"),
        ])
    def test_normalize_row(self, setup_solver, matrix, row_index, col_index, expected_matrix, expected_operation):
        solver = setup_solver(matrix)
        solver._GaussJordanSolver__normalize_row(row_index, col_index)
        assert solver.matrix == expected_matrix
        assert solver.operations[-1] == expected_operation

    def test_normalize_row_division_by_zero(self, setup_solver):
        solver = setup_solver([[0, 1, 4], [1, 2, 3], [3, 0, 2]])
        with pytest.raises(ZeroDivisionError):
            solver._GaussJordanSolver__normalize_row(0, 0)

    @pytest.mark.parametrize(
        "matrix, row_index, col_index, expected_matrix, expected_operations", [
            (
                [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                0, 0,
                [[Fraction(1, 1), Fraction(2, 1), Fraction(3, 1)],
                 [Fraction(0, 1), Fraction(-3, 1), Fraction(-6, 1)],
                 [Fraction(0, 1), Fraction(-6, 1), Fraction(-12, 1)]],
                ["F1-(4*F0)", "F2-(7*F0)"]
            ),
            (
                [[0, 2, 3], [4, 5, 6], [7, 8, 9]],
                1, 1,
                [[0, 2, 3], [4, 5, 6], [7, 8, 9]],
                []
            ),
            (
                [[0, 2, 3], [4, 1, 6], [7, 8, 9]],
                1, 1,
                [[Fraction(-8, 1), Fraction(0, 1), Fraction(-9, 1)],
                 [Fraction(4, 1), Fraction(1, 1), Fraction(6, 1)],
                 [Fraction(-25, 1), Fraction(0, 1), Fraction(-39, 1)]],
                ["F0-(2*F1)", "F2-(8*F1)"]
            ),
            (
                [], 1, 1, [], []
            ),
            (
                [[2, 1], [3, 2]], -2, 1, [[2, 1], [3, 2]], []
            )
        ])
    def test_make_colum_ceros(self, setup_solver, matrix, row_index, col_index, expected_matrix, expected_operations):
        solver = setup_solver(matrix)
        solver._GaussJordanSolver__make_colum_ceros(row_index, col_index)
        assert solver.matrix == expected_matrix
        assert solver.operations == expected_operations


    @pytest.mark.parametrize("vector, divider, expected_vector", [
            ([2, 4, 6], 2, [1, 2, 3]),
            ([1, 3, 0], 2, [0.5, 1.5, 0])
        ])
    def test_get_divided_vector(self,
                                simple_matrix, vector,
                                divider, expected_vector):

        assert simple_matrix._GaussJordanSolver__get_divided_vector(
            vector, divider) == expected_vector


    def test_get_divided_vector_zero_division(self, simple_matrix):
        with pytest.raises(ZeroDivisionError):
            simple_matrix._GaussJordanSolver__get_divided_vector([1, 2, 3], 0)

    @pytest.mark.parametrize("vector, expected_res", [
            ([0, 0, 0, 0], True),
            ([], True),
            ([0, 0, 3, 1], False)
        ])
    def test_all_ceros_vector(self, simple_matrix, vector, expected_res):
        assert simple_matrix._GaussJordanSolver__all_ceros(vector) is expected_res

    @pytest.mark.parametrize("matrix, expected_indexs", [
            ([[0, 1, 2], [1, 0, 0]], [0, 1]),
            ([[1, 2], [1, 0]], [0, 0]),
            ([[0, 0], [0, 0]], []),
            ([[0, 0, 0], [0, 0, 0], [0, 4, 0]], [2, 1]),
            ([[0, 2, 1], [0, 0, 2], [1, 0, 3], [0, 0, 0]], [0, 1])
        ])
    def test_get_not_null_num_index(self, setup_solver, matrix, expected_indexs):
        solver = setup_solver(matrix)
        assert solver._GaussJordanSolver__get_not_null_num_index() == expected_indexs

    @pytest.mark.parametrize("matrix, expected_matrix", [
            ([], []),
            ([[0, 0, 0], [0, 0, 0], [0, 0, 0]], [
             [0, 0, 0], [0, 0, 0], [0, 0, 0]]),  # Null matrix
            ([[1, 2, 3], [4, 5, 6], [7, 8, 9]], [
             [1, 2, 3], [4, 5, 6], [7, 8, 9]]),  # Not null matrix
            ([[0, 2, 1], [0, 0, 2], [1, 0, 3], [0, 0, 0]], [
             [0, 2, 1], [1, 0, 3], [0, 0, 2], [0, 0, 0]]),  # 4x3 matrix
            ([[0, 0, 0], [0, 1, 2], [1, 0, 0]],
             [[0, 1, 2], [1, 0, 0], [0, 0, 0]]),
            ([[0, 0], [1, 0], [0, 0]], [[1, 0], [0, 0], [0, 0]]),
            ([[0, 0, 1], [0, 0, 1]], [[0, 0, 1], [0, 0, 1]]),
            ([
                [Fraction(1, 1), Fraction(0, 2), Fraction(1, 1), Fraction(-2, 1), Fraction(3, 1), Fraction(0, 1), Fraction(78, 1)],
                [Fraction(0, 1),Fraction(0, 1),Fraction(0, 1),Fraction(0, 1),Fraction(0, 1),Fraction(0, 1),Fraction(0, 1)],
                [Fraction(0, 1),Fraction(0, 1),Fraction(0, 1),Fraction(0, 1),Fraction(0, 1),Fraction(0, 1),Fraction(-75, 1)],
                [Fraction(0, 1),Fraction(0, 1), Fraction(0, 1),Fraction(0, 1), Fraction(0, 1),Fraction(1, 1), Fraction(0, 1)],
                [Fraction(0, 1), Fraction(1, 1), Fraction(-2, 1), Fraction(3, 1), Fraction(-4, 1), Fraction(0, 1), Fraction(-47,1)],
                [Fraction(0, 1),Fraction(0, 1), Fraction(0, 1),Fraction(0, 1), Fraction(0, 1),Fraction(0, 1), Fraction(0, 1)]
            ],
            [
                [Fraction(1, 1), Fraction(0, 2), Fraction(1, 1), Fraction(-2, 1), Fraction(3, 1), Fraction(0, 1), Fraction(78, 1)],
                [Fraction(0, 1),Fraction(0, 1), Fraction(0, 1),Fraction(0, 1), Fraction(0, 1),Fraction(1, 1), Fraction(0, 1)],
                [Fraction(0, 1), Fraction(1, 1), Fraction(-2, 1), Fraction(3, 1), Fraction(-4, 1), Fraction(0, 1), Fraction(-47,1)],
                [Fraction(0, 1),Fraction(0, 1), Fraction(0, 1),Fraction(0, 1), Fraction(0, 1),Fraction(0, 1), Fraction(0, 1)],
                [Fraction(0, 1),Fraction(0, 1),Fraction(0, 1),Fraction(0, 1),Fraction(0, 1),Fraction(0, 1),Fraction(-75, 1)],
                [Fraction(0, 1),Fraction(0, 1),Fraction(0, 1),Fraction(0, 1),Fraction(0, 1),Fraction(0, 1),Fraction(0, 1)]
            ])
        ])
    def test_order_null_columns(self, setup_solver, matrix, expected_matrix):
        solver = setup_solver(matrix)
        solver._GaussJordanSolver__order_null_colums()
        assert solver.matrix == expected_matrix

    @pytest.mark.parametrize("matrix, expected_max", [
            ([[0, 0], [1, 222]], 4),
            ([[0, 1, 1], [1, 0, 0]], 3),
            ([], 0),
            ([[], []], 0),
            ([[11, 1, 1], [11, 22, 33], [33, 33, 444, 44]], 9)
        ])
    def test_get_max_char_amount_in_array(self, simple_matrix, matrix, expected_max):
        assert simple_matrix._GaussJordanSolver__get_max_char_amount_in_array(matrix) == expected_max
