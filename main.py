"""
    Main module to run program.
    Request matrix and instanciate database and solver.
"""
import argparse
import copy
import json
from src.gauss_jordan import GaussJordanSolver
from src.auxiliar_functions import is_matrix_range_inp_valid, convert_binary_lists_to_matrix
from src.db_manager import DataBaseManager

def main():
    parser = argparse.ArgumentParser(description="Gauss-Jordan Solver")
    parser.add_argument("-r", "--rows", type=int, required=True, help="Number of rows in the matrix")
    parser.add_argument("-c", "--cols", type=int, required=True, help="Number of columns in the matrix")
    args = parser.parse_args()
    if args.rows <= 0 or args.cols <= 0:
        parser.error("Error: Matrix dimensions must be greater than 0\n")

    matrix = []
    for i in range(args.rows):
        row = []
        for j in range(args.cols):
            while True:
                try:
                    input_val = float(input(f"Enter the element at position ({i},{j}): "))
                    row.append(input_val)
                    break
                except ValueError:
                    print("Invalid input. Please enter a valid number (int or float)")
        matrix.append(row)

    db = DataBaseManager()
    solver = GaussJordanSolver(matrix)
    matrix_copy = copy.deepcopy(matrix)
    solver.print_matrix()

    while True:
        correct = input("Is the matrix correct? (y/n): ").strip().lower()
        if correct == 'y':
            matrix_copy = copy.deepcopy(matrix)
            if db.check_exist(matrix_copy):
                print("Matrix already in database, searching solution...")
                results = db.get_solution(matrix_copy)[0]
                solver.operations = json.loads(results[0].decode('utf-8'))
                solver.implicit_solutions = json.loads(results[1].decode('utf-8'))
                solver.matrix = convert_binary_lists_to_matrix(results[2])
                solver.is_solved = True
            break

        if correct == 'n':
            try:
                row_index = int(input("Enter the row index of the element to change: "))
                col_index = int(input("Enter the column index of the element to change: "))
                # Check index range
                if not is_matrix_range_inp_valid(args.rows,args.cols,row_index,col_index):
                    continue
                
                new_value = float(input(f"Enter the new value for position ({row_index},{col_index}): "))
                solver.update_matrix_element(row_index, col_index, new_value)
                solver.print_matrix()
            except(ValueError):
                print("\nInvalid input. Please enter valid numbers (integers for indices, numbers for values).\n")
        else:
            print("Invalid response. Please enter 'y' or 'n'")

    if not solver.is_solved:
        solver.solve_matrix()
        db.save_solution(
        matrix_copy, solver.operations, solver.implicit_solutions, solver.matrix)

    solver.print_operations()
    solver.print_solutions()
    db.close()

if __name__ == "__main__":
    main()
