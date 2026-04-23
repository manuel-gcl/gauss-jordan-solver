# gauss-jordan-solver
## Description
This project provides a Python implementation of the Gauss-Jordan elimination method for solving systems of linear equations. 
The tool allows you to input a augmented matrix, validate and update it interactively, and then solve the system to find the solutions.
It implements a local SQLite database to store previously solved matrices and their operations, retrieving solutions to avoid resolving the same problem.

## Usage
1. **Execute the program:**
     ```python3 main.py -r <number_of_rows> -c <number_of_columns> ```
2. Input Matrix Values
3. Verify and Update Matrix
4. Solve the Matrix:
  The script will:
    1. Solve the matrix
    2. Print the matrix, operations performed, and the solutions
    3. Save matrix, made operations, implicit solution and resulting matrix in DB
  
## Requirements
+ Python 3.x
+ Numpy (only used for testing)
+ pytest
