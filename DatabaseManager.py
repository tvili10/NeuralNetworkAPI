import sqlite3
import numpy as np
class DatabaseManager:
    def __init__(self, db_name="mnist_data.db"):
        self.conn = sqlite3.connect(db_name)
        self.cursor = self.conn.cursor()
        self.create_table()

    def create_table(self):
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS mnist_training (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image BLOB,
                label INTEGER
            )
        ''')
        self.conn.commit()

    def insert_data(self, images, labels):
        data = [(img.tobytes(), lbl) for img, lbl in zip(images, labels)]
        self.cursor.executemany("INSERT INTO mnist_training (image, label) VALUES (?, ?)", data)
        self.conn.commit()

    def fetch_data(self):
        self.cursor.execute("SELECT image, label FROM mnist_training")
        rows = self.cursor.fetchall()
        images = [np.frombuffer(row[0], dtype=np.float32) for row in rows]
        labels = [row[1] for row in rows]
        return np.array(images), np.array(labels)

    def close_connection(self):
        self.conn.close()
