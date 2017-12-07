import unittest

import npdb

class TestDbarray(unittest.TestCase):
	def test_size(self):
		dbarr = npdb.dbarray((3,3,3), int)
		self.assertEqual(bdarr.size, 27)