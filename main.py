import argparse
from src.gauss_jordan import GaussJordanSolver

def main():
    parser = argparse.ArgumentParser(description="Gauss-Jordan Solver")
    parser.add_argument("-r", "--rows", type=int, required=True, help="Number of rows in the matrix")
    parser.add_argument("-c", "--cols", type=int, required=True, help="Number of columns in the matrix")
    args = parser.parse_args()

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

    solver = GaussJordanSolver(matrix)
    solver.print_matrix()


    while True:
        correct = input("Is the matrix correct? (y/n): ").strip().lower()
        if correct == 'y':
            break
        elif correct == 'n':
            row_index = int(input("Enter the row index of the element to change: "))
            col_index = int(input("Enter the column index of the element to change: "))
            new_value = float(input(f"Enter the new value for position ({row_index},{col_index}): "))
            solver.update_matrix_element(row_index, col_index, new_value)
            solver.print_matrix()
        else:
            print("Invalid response. Please enter 'y' or 'n'")
    
    solver.solve_matrix()
    solver.print_operations()
    solver.print_solutions()

if __name__ == "__main__":
    main()