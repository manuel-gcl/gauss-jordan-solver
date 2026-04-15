"""
    This module contains auxiliary functions used in conjunction with the Gauss-Jordan solver
"""
from fractions import Fraction
import string

def convert_matrix_to_fractions(matrix):
    """ Converts all elements of a matrix to their fractional representation """
    
    for index, row in enumerate(matrix):
        for sub_index, num in enumerate(row):
            matrix[index][sub_index] = get_fraction(num)
            
    return matrix
    
def get_fraction(number):
    """ Converts a number to its fractional representation """
    
    return Fraction(number).limit_denominator()

def get_vars_list(matrix):
    """ Generates a list of variable names based on the number of columns in the matrix """
    
    num_vars = len(matrix[0]) if matrix else 0
    letras = list(string.ascii_lowercase)

    return letras[:num_vars]

def is_matrix_range_inp_valid(m_rows, m_cols, inp_rows, inp_cols):
    if not (0 <= inp_rows < m_rows) or not (0 <= inp_cols < m_cols):
                    print(f"Invalid Input\n"
                    f"Row range: 0 to {m_rows - 1}\n"
                    f"Column range: 0 to {m_cols - 1}\n")
                    return False
    else:
        return True