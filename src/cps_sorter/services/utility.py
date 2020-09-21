import numpy as np
import os
import re

class Point:
    def __init__(self,x_init,y_init):
        self.x = x_init
        self.y = y_init

    def shift(self, x, y):
        self.x += x
        self.y += y

    def __repr__(self):
        return "".join(["Point(", str(self.x), ",", str(self.y), ")"])

    def __eq__(self, other):
        if (self.x == other.x) and (self.y == other.y):
            return True
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)


def get_distance(point_a, point_b):
    return np.sqrt( ((point_a.x-point_b.x)**2)+((point_a.y-point_b.y)**2))


def search_files(folder):
    beamng_pattern = re.compile(r".*beamng.*")
    deepdrive_pattern = re.compile(r".*deepdrive.*")
    driver_ai_pattern = re.compile(r".*driver_ai.*")
    # default_pattern = re.compile(r".*")
    file_pairs = {
        'beamng': {},
        'deepdrive': {},
        'driver_ai': {},
        'default': {}
    }
    for subdir, dirs, files in os.walk(folder):
        for filename in files:
            splited_subdir = subdir.split('\\')
            dir_name = splited_subdir[-1]
            filepath = subdir + os.sep + filename
            # if dir_name in ['execs', 'tests', 'final']:
            if beamng_pattern.match(filepath):
                file_pairs['beamng'].setdefault('{}-{}'.format(splited_subdir[-1],filename), {}).update({'exec_file': filepath})
            elif deepdrive_pattern.match(filepath):
                file_pairs['deepdrive'].setdefault('{}-{}'.format(splited_subdir[-1],filename), {}).update({'exec_file': filepath})
            elif driver_ai_pattern.match(filepath):
                file_pairs['driver_ai'].setdefault('{}-{}'.format(splited_subdir[-1],filename), {}).update({'exec_file': filepath})
            else:
                file_pairs['default'].setdefault('{}-{}'.format(splited_subdir[-1],filename), {}).update({'exec_file': filepath})

    return file_pairs
