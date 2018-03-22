import argparse

from classes.data_set_pickler import DataSetPickler
from mortality_prediction.mortality_data_preprocessor import MortalityDataPreprocessor


def main():
    parser = argparse.ArgumentParser(description="Script executes SQL query file to fetch mortality data "
                                                 "from queries catalog and saves pickled data in pickles catalog")
    parser.add_argument('queryFile', help='SQL Query File name in queries catalog to be executed')
    query_file = parser.parse_args().queryFile
    data_pickler = DataSetPickler(query_file, MortalityDataPreprocessor())
    data_pickler.preprocess_and_pickle_data_set()


if __name__ == "__main__":
    main()
