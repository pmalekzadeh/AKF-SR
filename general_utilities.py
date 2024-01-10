import csv
import errno
import os
import pathlib
import json

def ensure_directory_exists(base_directory):
    """
    Makes a directory if it does not exist
    """
    try:
        os.makedirs(base_directory)
    except OSError as ex:
        if ex.errno != errno.EEXIST:
            raise ex

def dump_dict_as_json(dict_to_dump, filename):
    p = pathlib.Path(filename)
    if len(p.parts) > 1:
        dump_dirs = pathlib.Path(*p.parts[:-1])
        ensure_directory_exists(str(dump_dirs))
    json_str = json.dumps(dict_to_dump)
    with open(filename, "w") as jsonfile:
        jsonfile.write(json_str)

class Time_Series_Statistics_Store(object):
    """
    Logs time series data.
    Header should represent every column in data.
    For example:
        epoch | loss
        0     | 1
        1     | 0.5
        2     | 0.3
    """
    def __init__(self, header):
        self.statistics = []
        self.header = header
    def add_statistics(self, data):
        if len(data) != len(self.header):
            raise ValueError("Data length does not match header")
        self.statistics.append(data)
    def dump(self, dump_filename="statistics.csv"):
        p = pathlib.Path(dump_filename)
        if len(p.parts) > 1:
            dump_dirs = pathlib.Path(*p.parts[:-1])
            ensure_directory_exists(str(dump_dirs))
        with open(dump_filename, "w") as csvfile:
            wr = csv.writer(csvfile)
            wr.writerow(self.header)
            for stat in self.statistics:
                wr.writerow(stat)
    def summarize_last(self):
        summary = ""
        for i in range(len(self.header)):
            if isinstance(self.statistics[-1][i], float):
                summary += "{}: {:.3f},".format(self.header[i], self.statistics[-1][i])
            else:
                summary += "{}: {},".format(self.header[i], self.statistics[-1][i])
        return summary
