# npdb: Large Parallelized NumPy Arrays
[![PyPI version](https://badge.fury.io/py/npdb.svg)](https://badge.fury.io/py/npdb)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Build Status](https://travis-ci.org/ryancwallace/npdb.svg?branch=master)](https://travis-ci.org/ryancwallace/npdb)
[![codecov](https://codecov.io/gh/ryancwallace/npdb/branch/master/graph/badge.svg)](https://codecov.io/gh/ryancwallace/npdb)

npdb is an implementation of large disk-stored NumPy-compatible n-dimenstional arrays that may exceed available memory. npdb implements the core multi-dimensional array class `npdb.dbarray`, which supports persistent binary storage and distributed batch processed operations. `npdb.dbarray` supports a subset of the `numpy.ndarray` interface.

## Background
The `numpy.memmap` class also supports arrays stored on disk but has several limitations. Arrays are stored in a single file, and the size of the file can not exceed 2GB on 32-bit systems. This implementation both restricts the size of data and disallows distributed storage. On the other hand, `numpy.memmap` objects support the entire `numpy.ndarray` interface. 

The npdb library strikes a different balance--array sizes are constrained only by available disk space and can be distributed across multiple files. The cost of this capability is that a limited subset of the numpy interface is supported.

## Example

```python
import npdb as nd

# create on disk a 3D array of floats of lengths 100
db_arr = npdb.dbarray((100,100,100), float)

# slice from array

```

## Installation
You can install npdb using pip by running

```
$ pip install npdb
```

## Testing
If unittest is installed, tests can be run after installation with

```
$ python -m unittest discover
```

## License
MIT License (see `LICENSE`). Copyright (c) 2017 Ryan Wallace.

## Authors
Ryan Wallace. ryanwallace@college.harvard.edu.


# IN DEV
TODO: 

---infrastruture---
* Sphinx
* Docs sight, links
* coverage tests
* compy numpy, cupy

---core content---
* set minimum size before spill over onto disk
* better handling of open(file) re exceptions,  remove file on delete
* magic methods 
* indexing location, module names?
* creation: basic - npdb.empty(), from data - npdb.array()
* free dbarray? parameter to save explicitly, or paramater for persistence? overload del
* keep track of what's in memory to avoid repetitive pulls
* default size params, data dir
* rewrite arraymap and indexing in Cython/C/numpy?
* compression
