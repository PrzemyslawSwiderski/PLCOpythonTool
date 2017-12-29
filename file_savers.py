import csv
import json


class FileSaver:
    def save_to_file(self, output_file, list_to_save):
        pass


class CsvSaver(FileSaver):
    def save_to_file(self, output_file, list_to_save):
        writer = csv.writer(output_file, delimiter=',')
        writer.writerow(list_to_save[0].keys())
        for row in list_to_save:
            writer.writerow(row.values())


class JsonSaver(FileSaver):
    def __init__(self, indent=None, sort_keys=False):
        self.indent = indent
        self.sort_keys = sort_keys

    def save_to_file(self, output_file, list_to_save):
        part_str = json.dumps(list_to_save, indent=self.indent, sort_keys=self.sort_keys)
        output_file.write(f"{part_str}\n")
