"""
Contains the two core npdb classes: 
    1) dbarray, an ndarray-like object stored on disk, and
    2) dbview, a subclass of ndarray that corresponds to a subsection
    of a dbarray
"""

import os
import operator
import itertools

import numpy as np

import npdb
from npdb.core.diskmap import arraymap

class dbarray(object):
    """
    Implements a ndarray-like object with data stored on disk, not memory. Data
    size is only constrained by available space on disk.
    """
    def __init__(self, shape, dtype, buff=None, offset=None, strides=None,
                 order="C", data_dir=None, max_file_size=None):
        # items
        self.dtype = np.dtype(dtype)

        # size attributes
        if shape is None:
            self.shape = ()
        elif isinstance(shape, int):
            # convert to length 1 tuple
            self.shape = (shape,)
        else:
            self.shape = tuple(shape)
        assert(all([s >= 0 for s in self.shape]))

        self.size = reduce(operator.mul, self.shape)
        self.ndim = len(shape)

        # calculate disk C-contiguous strides; data always stored C-contiguous
        strides = [1] * self.ndim
        for axis in range(self.ndim - 1):
            strides[axis + 1] = strides[axis] * self.shape[axis]
        self.strides = reversed(strides)

        # maximum allowed file size in bytes
        self.max_file_size = max_file_size

        # allocate disk space and map array contents to disk space 
        try: 
            self.arrmap = arraymap(shape, dtype, strides, data_dir,
                                   max_file_size)
        except Exception as e:
            raise RuntimeError, "Disk allocation failed. {}".format(e)

        # write data to disk 
        if not buff is None:
            arr = np.ndarray(shape, dtype, buff, offset, strides, order)
            self.write(arr, raw_idx=None)

    def __repr__(self):
        return "{} dbarray of {}".format(self.shape, self.dtype, 
                                         self.arrmap.array_dir_name)

    def __len__(self):
        """
        Following NumPy convention, len(dbarray) is the size of the first
        dimension.
        """
        return self.shape[0]

    def __getitem__(self, raw_idx):
        """
        Overloads access indexing.
        """
        return self.read(raw_idx)

    def __setitem__(self, raw_idx, arr):
        """
        Overloads assignment indexing.
        """
        self.write(arr, raw_idx)

    def __del__(self, arr):
        """
        Overloads del keyword. 
        """
        # delete files on disk
        print "IN del", arr
        os.removedirs(arr.dir_name)

    def read(self, raw_idx=None):
        """
        Returns (in-memory) dbview object corresponding to dbarray[raw_idx].
        """
        # read data from disk contained in bounding indices
        indexed_shape, index_bounds = indexing.unravel_index(self.shape, raw_idx)
        data = self.arrmap.pull(index_bounds, indexed_shape)
        
        # create dbview
        view = dbview(data, self, index_bounds)

        return view

    def write(self, arr, raw_idx=None):
        """
        Pushes data in arr to dbarray on disk.
        """
        # find bounding indices and push between them
        indexed_shape, index_bounds = npdb.indexing.unravel_index(self.shape, 
                                                                  raw_idx)
        self.arrmap.push(index_bounds, arr)

    def asndarray(self, copy):
        """
        Returns view of entire dbarray as ndarray object.
        """
        full_view = self.read()
        return full_view.asndarray(copy)

class dbview(np.ndarray):
    def __new__(cls, data, dbarray, arrslice=None):
        """
        Describes a "view" into a subset of a dbarray. Inherits from np.ndarray; 
        additional parameters locate the view in containing dbarray. The position of
        an dbview in a dbarray is described by its dbindex object arrslice.
        """
        obj = np.asarray(data).view(cls)
        
        # parent
        obj.dbarray = dbarray

        # location in parent
        obj.arrslice = arrslice
        
        return obj

    def __repr__(self):
        pass

    def __array_finalize__(self, obj):
        if obj is None: return
        self.dbarray = getattr(obj, 'dbarray', None)
        self.arrslice = getattr(obj, 'arrslice', None)

    def flush(self):
        """
        Pushes current data in view to disk. 
        """
        self.dbarray.flush(self)

        return None

    def asndarray(self, copy=False):
        """
        Casts dbview as an np.ndarray, obliterating arrslice context. If copy, data are copied and
        pointer to new ndarray object is returned; otherwise, points to original dbview memory. 
        """
        if copy:
            arr = np.array(self)
        else:
            arr = np.asarray(self)

        return arr
