from setuptools import setup, find_packages

# manual package listing
# packages = ['npdb',
#             'npdb.core',
#             'npdb.creation',
#             'npdb.indexing',
#             'npdb.linalg',
#             'npdb.manipulation',
#             'npdb.math',
#             'npdb.random', 
#             'npdb.sorting',
#             'npdb.statistics']

# and automatic
packages = find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"])

setup(name='npdb',
      version='0.1',
      description='npdb: Parallel NumPy-like interface for large n-dimensional arrays on disk.',
      url='https://github.com/ryancwallace/npdb',
      author='Ryan Wallace',
      author_email='ryanwallace@college.harvard.edu',
      license='MIT',
      packages=packages,
      zip_safe=False)