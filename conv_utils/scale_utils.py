import os
import common
import argparse
import numpy as np

class Scale:
    """
    Scales a bunch of meshes.
    """

    def __init__(self, options):
        """
        Constructor.
        """
        self.options = options

    def read_directory(self, directory):
        """
        Read directory.

        :param directory: path to directory
        :return: list of files
        """

        files = []
        for filename in os.listdir(directory):
            files.append(os.path.normpath(os.path.join(directory, filename)))

        return files

if __name__ == '__main__':
    app = Scale()
    app.run()