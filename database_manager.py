import sqlite3
import numpy as np

name_of_database = "user_trainingdata.db"

class Database_Manager:
    def __init__(self):
        self.conn = sqlite3.connect(name_of_database, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self.create_table()

    def create_table(self):
        self.cursor.execute('''
                    CREATE TABLE IF NOT EXISTS training_data (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            pixels BLOB,
                            label INTEGER
                    )
                    ''')
        self.conn.commit()

    def add_data(self, pixels, label):
        pixels_blob = np.array(pixels, dtype=np.float32).tobytes()
        self.cursor.execute('''
                INSERT INTO training_data (pixels, label)
                VALUES (?, ?)
                ''', (pixels_blob, label))
        self.conn.commit()
