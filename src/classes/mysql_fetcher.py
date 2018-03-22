import logging

import pandas
import pymysql

from classes.query_loader import QueryLoader


class MySqlFetcher:
    def __init__(self, query_loader=QueryLoader()):
        self.__db_connection = self.open_connection()
        self.data_set = pandas.DataFrame()
        self.query_loader = query_loader

    def open_connection(self):
        return pymysql.connect(host='localhost',
                               user='root',
                               password='root',
                               db='prostate_screening',
                               charset='utf8mb4',
                               cursorclass=pymysql.cursors.DictCursor)

    def close_connection(self):
        self.__db_connection.close()

    def run_select_query(self, query, arguments=None):
        self.data_set = pandas.read_sql_query(query, self.__db_connection, params=arguments)
        logging.info(f"\nFetched data from DB: ")
        logging.info(f"\n{self.data_set}")
        return self.data_set

    def run_select_query_from_file(self, file_name):
        query = self.query_loader.load_query(file_name)
        return self.run_select_query(query)
