import pandas
import pymysql
from sklearn.preprocessing import MinMaxScaler

from details_printer import DefaultDetailsPrinter
from query_loader import QueryLoader


class MySqlFetcher:
    def __init__(self, scaler=MinMaxScaler(), query_loader=QueryLoader()):
        self.__db_connection = self.open_connection()
        self.data_set = pandas.DataFrame()
        self.data_set_normalized = pandas.DataFrame()
        self.scaler = scaler
        self.query_loader = query_loader
        self.details_printer = DefaultDetailsPrinter(self)

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
        query_result = pandas.read_sql_query(query, self.__db_connection, params=arguments)
        query_result = query_result.apply(pandas.to_numeric)
        self.data_set = query_result
        self.normalize_data_set()
        return self.data_set

    def normalize_data_set(self):
        self.data_set_normalized = pandas.DataFrame(self.scaler.fit_transform(self.data_set.iloc[:, 0:-1]))

    def get_X(self):
        return self.data_set_normalized.values

    def get_Y(self):
        return self.data_set.values[:, -1]

    def run_select_query_from_file(self, file_name):
        query = self.query_loader.load_query(file_name)
        self.run_select_query(query)
