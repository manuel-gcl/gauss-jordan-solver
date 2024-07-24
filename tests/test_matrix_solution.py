"""
This module test gauss_jordan algorithm against other 
resolution of linear systems obtained 
from https://medium.com/jungletronics/linear-equations-solve-by-gauss-jordan-49c3c8474173

"""
from random import randrange
import numpy as np

from src.gauss_jordan import GaussJordanSolver


class TestSolverResults:
    def __init__(self):
        self.matrix_set = []

    def _generate_matrix_d(self, amount=10, columns=None, rows=None):
        while amount >= 1:
            col = randrange(1, 10) if columns is None else columns
            row = randrange(1, 10) if rows is None else rows
            matrix = np.random.randint(100, size=(row, col))
            self.matrix_set.append(np.ndarray.tolist(matrix))
            amount -= 1
    
    def get_coefficient_and_dependent_matrix(self, matrix):
        coefficient_matrix = []
        dependent_matrix = []
        for row in matrix: 
            coefficient_matrix.append(row[:-1]) 
            dependent_matrix.append(row[-1])
        
        return coefficient_matrix, dependent_matrix
    
    def get_res_with_gauss_jordan(self, matrix):
        solver = GaussJordanSolver(matrix)
        solver.solve_matrix()
        _, dep_matrix = self.get_coefficient_and_dependent_matrix(solver.matrix)
        dep_matrix = [float(num) for num in dep_matrix]

        return dep_matrix
    
    def get_res_with_numpy(self, coef_mat, dep_mat):
        inv_coef_mat = np.linalg.inv(coef_mat)
        res = np.matmul(inv_coef_mat, dep_mat)
        res = np.ndarray.tolist(res)
            
        return res
    
    def compare_results(self, matrixA, matrixB):
        res = True
        matrixA.sort()
        matrixB.sort()
        
        for index, num in enumerate(matrixA):
            if round(num, 5) != round(matrixB[index], 5):
                res = False
                break

        return res
    
if __name__ == "__main__":
    d = TestSolverResults()
    d._generate_matrix_d(100, 5, 4)
    
    for m in d.matrix_set:
        coef_m, dep_m = d.get_coefficient_and_dependent_matrix(m)
        np_res = d.get_res_with_numpy(np.array(coef_m), np.array(dep_m))

        gaussJordan_res = d.get_res_with_gauss_jordan(m)

        res = d.compare_results(np_res, gaussJordan_res)
        if not res:
            print("Square Matrix has different results\n")
            print(f"GJ P Method: {gaussJordan_res}\n")
            print(f"Numpy Method {np_res}\n")
    print("End")
