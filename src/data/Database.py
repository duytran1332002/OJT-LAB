import psycopg2

class Database:
    def __init__(self, host = "localhost", database = "OJT_LAB_Q2", user = "postgres", password = "adidaphat"):
        self.connection = psycopg2.connect(
            host=host,
            database=database,
            user=user,
            password=password
        )
        self.cursor = self.connection.cursor()
    

    def insert_values(self, table_name, attribute, values):
        """Inserts values into the table
        Args:
            table_name (str): name of the table
            values (list): list of values to be inserted
        """
        try:
            self.cursor.execute(f"""INSERT INTO {table_name} ({attribute}) VALUES (%s, %s, %s)""", values)
            self.connection.commit()
        except Exception as e:
            print(e)
            self.connection.rollback()
    
    def delete_values(self, table_name, condition):
        """Deletes values from the table
        Args:
            table_name (str): name of the table
            condition (str): condition to be met
        """
        try:
            self.cursor.execute(f"""DELETE FROM {table_name} WHERE {condition}""")
            self.connection.commit()
        except Exception as e:
            print(e)
            self.connection.rollback()

    def get_value(self, table_name, condition):
        """Gets values from the table
        Args:
            table_name (str): name of the table
            condition (str): condition to be met
        """
        try:
            self.cursor.execute(f"""SELECT * FROM {table_name} WHERE {condition}""")
            return self.cursor.fetchall()
        except Exception as e:
            print(e)
            self.connection.rollback()
    
    def get_all_values(self, table_name):
        """Gets all values from the table
        Args:
            table_name (str): name of the table
        """
        try:
            self.cursor.execute(f"""SELECT * FROM {table_name}""")
            return self.cursor.fetchall()
        except Exception as e:
            print(e)
            self.connection.rollback()
    
    def implement_find_similar_passage(self, question_embedding, k = 5):
        """Gets all values from the table
        Args:
            table_name (str): name of the table
        """
        try:
            self.cursor.execute("""SELECT * FROM find_similar_passages(CAST(%s AS vector(768)), %s)""", (question_embedding.tolist(), k))
            return self.cursor.fetchall()
        except Exception as e:
            print(e)
            self.connection.rollback()
    