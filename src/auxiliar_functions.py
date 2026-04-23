"""
    This module contains auxiliary functions used in conjunction with the Gauss-Jordan solver
"""
import hashlib
from fractions import Fraction
import json
import string

def convert_matrix_to_fractions(matrix):
    """ Converts all elements of a matrix to their fractional representation """
    
    for index, row in enumerate(matrix):
        for sub_index, num in enumerate(row):
            matrix[index][sub_index] = get_fraction(num)
            
    return matrix

def convert_nums_list_to_string(nums_lists):
    
    return [[str(num) for num in row] for row in nums_lists]

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
        "Column range: 0 to {m_cols - 1}\n")
        return False
    return True

def convert_list_to_json(list_obj):
    return json.dumps(list_obj).encode('utf-8')

def get_matrix_hash(matrix):
    m_hash = hashlib.sha256()
    m_hash.update(get_json_object(matrix))
        
    return m_hash

def get_json_object(data):
    if isinstance(data, bytes):
        return data
    return convert_list_to_json(convert_nums_list_to_string(data))

def convert_binary_lists_to_matrix(json_data):
    data = json_data
    if isinstance(data, bytes):
        data = json.loads(data.decode('utf-8'))
 
    converted_matrix = []
    for row in data:
        new_row = []
        for item in row:
            if isinstance(item, str):
                new_row.append(get_fraction(item))
            else:
                new_row.append(item)
        converted_matrix.append(new_row)
 
    return converted_matrix