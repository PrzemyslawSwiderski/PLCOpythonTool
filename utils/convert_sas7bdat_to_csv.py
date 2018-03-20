#!/usr/bin/python

import sys

import os
from sas7bdat import SAS7BDAT


def main(argv):
    if len(argv) > 0:
        file_path = os.path.abspath(argv[0])
        dir_name = os.path.dirname(file_path)
        base_name = os.path.basename(file_path)
        file_name = os.path.splitext(base_name)[0]
        out_put_file_path = os.path.join(dir_name, f"{file_name}.csv")

        with SAS7BDAT(f"{file_path}") as sas7bdat_file:
            sas7bdat_file.convert_file(out_put_file_path)
    else:
        print("Please pass file_path as 1. script's argument.")
        print("For example: 'convert_sas7bdat_to_csv.py ../freepsa_data_feb16_d080516.sas7bdat'")


if __name__ == "__main__":
    main(sys.argv[1:])
