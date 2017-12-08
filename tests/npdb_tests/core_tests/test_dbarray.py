import unittest

import numpy as np

import npdb

class TestDbarray(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dbarr = npdb.dbarray((3,3,3), int)

    def test_size(self):
        self.assertEqual(self.dbarr.size, 27)

    def test_len(self):
        self.assertEqual(len(self.dbarr), 3)

    def test_set(self):
        self.dbarr[1]

    @classmethod
    def tearDownClass(cls):
        pass