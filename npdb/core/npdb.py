"""
Contains the two core npdb classes: 
    1) dbarray, an ndarray-like object stored on disk, and
    2) dbview, a subclass of ndarray that corresponds to a subsection
    of a dbarray
"""

import sys
import operator
import itertools

import numpy as np

import indexing
import arraymap as am

class dbarray(object):
    """
    Implements a ndarray-like object with data stored on disk, not memory. Data
    size is only constrained by available space on disk.
    """
    def __init__(self, shape, dtype, byteorder="big_endian", 
                 order="C", data_dir=None, max_file_size=None):
        # size attributes
        if type(shape) is int:
            # convert to length 1 tuple
            self.shape = (shape,)
        else:
            self.shape = tuple(shape)
        self.size = reduce(operator.mul, self.shape)
        self.ndim = len(shape)

        self.dtype = np.dtype(dtype)
        self.itemsize = self.dtype.itemsize

        self.byteorder = byteorder
        self.order = order

        self.max_file_size = max_file_size

        # allocate disk space and map array contents to disk space 
        try:
            self.arrmap = am.arraymap(shape, dtype, data_dir, max_file_size)
        except Exception as e:
            print "Disk allocation failed.", e
            sys.exit(e)

    def __repr__(self):
        pass

    def __len__(self):
        """
        Following NumPy convention, len(dbarray) is the size of the first dimension.
        """
        return self.shape[0]

    def __getitem__(self, idx):
        """
        Overload access indexing.
        """
        bounds = indexing.index(self.shape, idx)
        return bounds
        # return self.read(arrslice)

    def __setitem__(self, idx, dbview):
        """
        Overload assignment indexing.
        """
        arrslice = indexing.index(self.shape, idx)
        # view = dbview(dbview.asndarray(), dbview.dbarray, arrslice=arrslice)

        # self.flush(view)

    def read(self, arrslice=None):
        """
        Returns (in-memory) dbview object corresponding to dbarray[arrslice].
        """
        # read data from disk contained in bounding indices
        flattened_bounds = indexing.dbindex.flattened_bounds(arrslice)
        shape = indexing.dbindex.shape(arrslice)
        data = self.am.pull(flattened_bounds, shape)
        
        # create dbview
        view = dbview(data, self, arrslice)

        return view

    def flush(self, dbview):
        """
        Pushes data in dbview to dbarray on disk.
        """
        # convert arrslice to flattened bounds
        # flattened_bounds = dbindex.flattened_bounds(arrslice, self)

        # map copies ndarray to disk
        self.am.push(dbview)

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
