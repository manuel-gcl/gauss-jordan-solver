"""
This module contains tests for the auxiliary functions used in the Gauss-Jordan solver
"""
from fractions import Fraction
import pytest

from src.auxiliar_functions import convert_matrix_to_fractions, get_fraction, get_vars_list

class TestAuxiliarFunctions:
    @pytest.mark.parametrize("matrix, expected_matrix", [
            ([[0.5, 0.25], [1.5, 2.5]], [
             [Fraction(1, 2), Fraction(1, 4)], [Fraction(3, 2), Fraction(5, 2)]]),
            ([[0, 1], [2, -3]], [[Fraction(0), Fraction(1)], [Fraction(2), Fraction(-3)]]),
            ([[0.333333], [-2.5]], [[Fraction(333333, 1000000)], [Fraction(-5, 2)]]),
            ([[]], [[]]),  # Empty matrix
            ([], []),  # Empty matrix
            ([[1/3, 1/7], [1/9, 1/11]], [[Fraction(1, 3), Fraction(1, 7)],
             [Fraction(1, 9), Fraction(1, 11)]]),
            ([[Fraction(0), Fraction(1)]], [[Fraction(0), Fraction(1)]])
        ])
    def test_convert_matrix_to_fractions(self, matrix, expected_matrix):
        assert convert_matrix_to_fractions(matrix) == expected_matrix

    @pytest.mark.parametrize("number, expected_fraction", [
            (0.5, Fraction(1, 2)),
            (1.5, Fraction(3, 2)),
            (0.333333, Fraction(333333, 1000000)),
            (-2.5, Fraction(-5, 2)),
            (0, Fraction(0)),
            (2, Fraction(2)),
            (1/3, Fraction(1, 3)),
            (1/7, Fraction(1, 7))
        ])
    def test_get_fraction(self, number, expected_fraction):
        assert get_fraction(number) == expected_fraction

    @pytest.mark.parametrize("matrix, expected", [
        ([], []),
        ([[1]], ["a"]),
        ([[1], [2]], ["a"]),
        ([[1], [2], [3], [4], [5], [6], [7], [8]], ["a"]),
        ([[1, 3, 4], [2, 4, 5], [3, 4, -1]], ["a", "b", "c"]),
        ([[1, 2, 3, 4], [3, 3, 3, 3]], ["a", "b", "c", "d"])
    ])
    def test_get_vars_list(self, matrix, expected):
        assert get_vars_list(matrix) == expected
