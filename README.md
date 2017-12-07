# IN DEV
TODO: 
---infrastruture---
imports into top namespace
testing
Sphinx
packaging

---core content---
creation
* basic - npdb.empty()
* from data - npdb.array()
rewrite arraymap and indexing in Cython/C/numpy?
free dbarray? parameter to save explicitly, or paramater for persistence?
keep track of what's in memory to avoid repetitive pulls
compression

# npdb

npdb is an implementation of large disk-stored NumPy-compatible n-dimenstional arrays that may exceed available memory. npdb implements the core multi-dimensional array class `npdb.dbarray`, which supports persistent binary storage and distributed batch processed operations. `npdb.dbarray` supports a subset of the `numpy.ndarray` interface.

## Background
The `numpy.memmap` class also supports arrays stored on disk, but has several limitations. Arrays are stored in a single file, and the size of the file can not exceed 2GB on 32-bit systems. This restricts the size of data, and does not allow for distributed storage. On the other hand, `numpy.memmap` objects support all numpy operations. 

The npdb library strikes a different balance--array sizes are constrained only by available disk space and can be distributed across multiple disks. The cost of this capability is that a limited subset of the numpy interface is supported.

## Example
```
import npdb as nd

# create on disk a 3D array of floats of lengths 100
db_arr = npdb.dbarray((100,100,100), float)

# slice from array

```

## Installation

## Testing

## License
MIT License (see `LICENSE`). Copyright (c) 2017 Ryan Wallace.

## Authors
Ryan Wallace. ryanwallace@college.harvard.edu.