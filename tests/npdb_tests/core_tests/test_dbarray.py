import unittest

import numpy as np

import npdb

class TestDbarray(unittest.TestCase):
	def test_size(self):
		dbarr = npdb.dbarray((3,3,3), int)
		self.assertEqual(dbarr.size, 27)
