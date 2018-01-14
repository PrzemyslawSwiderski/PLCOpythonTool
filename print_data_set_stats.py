import logging
import logging.config

from mysql_fetcher import MySqlFetcher


def main():
    logging.config.fileConfig("logging.conf")

    ms = MySqlFetcher()
    ms.run_select_query_from_file("queries/select_to_predict_mortality_2.sql")
    ms.details_printer.print_all()


if __name__ == "__main__":
    main()
