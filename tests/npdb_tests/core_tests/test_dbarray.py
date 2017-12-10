import unittest

import numpy as np

import npdb

class TestDbarray(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dbarr = npdb.dbarray((3,3,3), int)
        cls.ndarr = np.arange(27).reshape((3,3,3))

    def test_size(self):
        self.assertEqual(self.dbarr.size, 27)

    def test_len(self):
        self.assertEqual(len(self.dbarr), 3)

    def test_set(self):
        self.dbarr[:] = self.ndarr

    # def test_delete(self):
    #     del self.dbarr

    @classmethod
    def tearDownClass(cls):
        # del self.ndarr
        pass
