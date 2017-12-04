"""
Two core npdb classes: 
    1) dbarray, an ndarray-like object stored on disk, and
    2) dbview, a subclass of ndarray that corresponds to a subsection
    of a dbarray
"""

import sys
import operator
import itertools

import numpy as np

import arraymap as am

class dbarray(object):
    """
    Implements a ndarray-like object with data stored on disk, not memory. Data size
    is only constrained by available space on disk.
    """
    def __init__(self, shape, dtype, data_dir=None, max_chunk_size=None):
        # size attributes
        self.shape = tuple(shape)
        self.size = reduce(operator.mul, self.shape)
        self.ndim = len(shape)
        self.max_chunk_size = max_chunk_size

        # allocate disk space and map array contents to disk space 
        try:
            self.arrmap = am.arraymap(shape, dtype, data_dir, max_chunk_size)
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

    def __getitem__(self, slc):
        """
        Overload access indexing.
        """
        arrslice = dbindex(slc)
        return self.read(arrslice)

    def __setitem__(self, slc, ndview):
        """
        Overload assignment indexing.
        """
        arrslice = dbindex(slc)
        view = ndview(ndview.asndarray(), ndview.dbarray, arrslice=arrslice)

        self.flush(view)

    def read(self, arrslice=None):
        """
        Returns (in-memory) ndview object corresponding to dbarray[arrslice].
        """
        # read data from disk contained in bounding indices
        flattened_bounds = dbindex.flattened_bounds(arrslice)
        shape = dbindex.shape(arrslice)
        data = self.am.pull(flattened_bounds, shape)
        
        # create ndview
        view = ndview(data, self, arrslice)

        return view

    def flush(self, ndview):
        """
        Pushes data in ndview to dbarray on disk.
        """
        # convert arrslice to flattened bounds
        flattened_bounds = dbindex.flattened_bounds(arrslice, self)

        # map copies ndarray to disk
        self.am.push(ndview.asndarray(), flattened_bounds)

    @staticmethod
    def asndarray(self, dbarray, copy):
        """
        Returns view of entire dbarray as ndarray object.
        """
        full_view = self.read()
        return full_view.asndarray(copy)

class ndview(np.ndarray):
    def __new__(cls, data, dbarray, arrslice=None):
        """
        Describes a "view" into a subset of a dbarray. Inherits from np.ndarray; 
        additional parameters locate the view in containing dbarray.
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
        Casts ndview as an np.ndarray, obliterating bounds. If copy, data are copied and
        pointer to new ndarray object is returned; otherwise, points to original ndview memory. 
        """
        if copy:
            arr = np.array(self)
        else: 
            arr = np.asarray(self)

        return arr 

    @staticmethod
    def merge(self, ndviews, fill=0):
        """
        Uses ndview positions to merge multiple ndview objects into a single new ndview object.

        TODO: fix merged_dims to account for parallel slices. Find bounding hypercube.
        """
        # parent_ndim = ndviews[0].dbarray.ndim

        # # check that all ndviews have same dtype
        # dtypes = [view.dtype for view in ndviews]
        # assert(all(dtypes[0] == dtype for dtype in dtypes)), "All ndview dtypes must be equal"

        # # calculate dims and offsets of merged array
        # merged_dims = list(set(sorted(itertools.chain.from_iterable([view.dims for view in ndviews]))))
        # offsets_by_dim = [[view.offset[dim] for view in ndviews] for dim in range(parent_ndim)]
        # offsets_by_view = [[offsets[view_idx] for offsets in offsets_by_dim] for view_idx in range(len(ndviews))]
        # merged_offset = [min(offsets) for offsets in offsets_by_dim]

        # # calculate position of merged view in dbarray
        # length_by_view = [[view.shape[view.dims.index(dim)] if dim in view.dims else 1 for view in ndviews] for dim in range(parent_ndim)]
        # upper_bounds = [max([view.offset[dim] + length_by_view[dim][view_idx] for view_idx, view in enumerate(ndviews)]) for dim in range(parent_ndim)]
        # merged_bounds = zip(merged_offset, upper_bounds)
        # length_by_dim =  [u - l for l, u in merged_bounds] 
        # length_by_merged_dim = [l for l in length_by_dim if l != 1]
        # flattened = [l == 1 for l in length_by_dim]
        # flattened_dims = list(itertools.compress(range(parent_ndim), flattened))

        # # create new empty ndview object
        # if fill == 0:
        #     merged_arr = np.zeros(shape=length_by_merged_dim, dtype=dtypes[0])
        # elif fill == 1:
        #     merged_arr = np.ones(shape=length_by_merged_dim, dtype=dtypes[0])
        # else:
        #     merged_arr = np.empty(shape=length_by_merged_dim, dtype=dtypes[0])
        #     merged_arr[:] = fill

        # merged = ndview(merged_arr, ndviews[0].dbarray, dims=merged_dims, offset=merged_offset)
        
        # print parent_ndim
        # print merged_dims
        # print offsets_by_dim
        # print offsets_by_view
        # print merged_offset
        # print 
        # print length_by_view
        # print upper_bounds
        # print merged_bounds
        # print length_by_dim
        # print length_by_merged_dim
        # print
        # print merged

        # for view_idx, view in enumerate(ndviews): 
        #     offset = offsets_by_view[view_idx]
        #     print "view_idx", view_idx
        #     print "dims", view.dims
        #     print "offset", offset
        #     for idx, val in np.ndenumerate(view):
        #         idx = list(idx)
        #         idx_full = [idx.pop(0) if d in view.dims else 0 for d in range(len(offset))]
        #         idx_offset = [sum(i) for i in zip(idx_full, offset)]
        #         print "assign", idx_offset, val
        #         merged[tuple(idx_offset)] = val

        # print 
        # return merged

        # # find bounding hypercube: the size of merged array
        # hypercube = _bounding_hypercube(ndviews)

        # # find offset of merged array relative to containing dbarray
        # merged_offset = 



db = dbarray((3,3,3), float, "file")
s = db[2,2:]

# a = ndview(np.zeros(shape=(3,3)), db, dims=(0,1), offset=(0,0,0))
# b = ndview(np.ones(shape=(2,2)), db, dims=(0,1), offset=(1,0,0))
# c = ndview([[3,1],[1,5],[2,3]], offset=(1,2))
# d = ndview([[3,1],[1,5]], offset=(6,9))
# print a
# print b

# print ndview.merge([a,b], fill=2)

# c = a.asndarray(copy=True)
# c[0,0] = 100
# print c
# print a
# print type(c)
# print type(a)




