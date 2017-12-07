from setuptools import setup

packages = ['npdb',
  			'npdb.core',
  			'npdb.creation',
  			'npdb.indexing',
  			'npdb.linalg',
  			'npdb.manipulation',
  			'npdb.math',
  			'npdb.random', 
  			'npdb.sorting',
  			'npdb.statistics']

setup(name='npdb',
      version='0.1',
      description='npdb: NumPy-compatible large, persistent n-dimensional arrays on disk',
      url='https://github.com/ryanwallace96/npdb',
      author='Ryan Wallace',
      author_email='ryanwallace@college.harvard.edu',
      license='MIT',
      packages=packages,
      zip_safe=False)