"""
    Module creates database in current project path
    Saves Matrix, operations and results in DB.
"""
from pathlib import Path
import sqlite3
from src.auxiliar_functions import get_matrix_hash, convert_list_to_json, get_json_object

DB_FILENAME = "matrix_history.db"
DB_PATH = Path(__file__).resolve().parent.parent / DB_FILENAME

class DataBaseManager:
    def __init__(self):        
        self.db_path = DB_PATH
        self.connection = sqlite3.connect(str(self.db_path))
        self.cursor = self.connection.cursor()
        self._create_table()

    def close(self):
        self.connection.close()
    
    def _create_table(self):
        query = """
            CREATE TABLE IF NOT EXISTS matrix_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                hash_id TEXT UNIQUE NOT NULL,
                matrix_data TEXT NOT NULL,
                operations TEXT NOT NULL,
                solution_data TEXT NOT NULL,
                end_matrix TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
        self.cursor.execute(query)
        self.connection.commit()
           
    def save_solution(self, org_matrix, operations, solution, end_matrix):
        """
        Runs query to save values in JSON format in DB
        """
        query = """
        INSERT INTO matrix_history (
            hash_id, 
            matrix_data, 
            operations, 
            solution_data, 
            end_matrix)
        VALUES (?, ?, ?, ?, ?);"""
        
        try:
            self.cursor.execute(query, (
                get_matrix_hash(org_matrix).hexdigest(),
                get_json_object(org_matrix),
                convert_list_to_json(operations),
                convert_list_to_json(solution),
                get_json_object(end_matrix)))

            self.connection.commit()
            print("Results saved in database succesfully")
        except sqlite3.IntegrityError as e:
            print(f"¡Changes not saved! Integrity Error: {e}")
            self.connection.rollback()
        except sqlite3.Error as e:
            print(f"Database Error: {e}")
            self.connection.rollback()

    def check_exist(self, matrix):
        """
            Verifiy existence of matrix in DB
        """
        query = """
        SELECT EXISTS(SELECT 1 FROM matrix_history WHERE hash_id = ?);
        """
        self.cursor.execute(
            query, 
            (get_matrix_hash(matrix).hexdigest(),)
            )
       
        return self.cursor.fetchone()[0]
   
    def get_solution(self, matrix):
        """
            Returns matrix information from DB
        """
        query = """
        SELECT operations, solution_data, end_matrix 
        FROM matrix_history WHERE hash_id = ?;
        """
        self.cursor.execute(
            query,
            (get_matrix_hash(matrix).hexdigest(),)
            )
        
        return self.cursor.fetchall()
