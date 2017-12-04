import sys
import operator
import itertools

import numpy as np

import arraymap as am

class ndslice(object):
    """
    An ndslice is a one of a slice, an int, or a tuple of slices and ints.
    """
    def __init__(self, slc):
        """
        Stores slice and loosely checks that the type is workable.
        """        
        # try to cast slc as a workable type
        if not type(slc) in [slice, int, tuple]:
            try:
                slc = slice(slc)
            except TypeError as e:
                print "ndslice must be one of a slice, an int, or a tuple of slices and ints", e
                sys.exit(e)

        self.arrslice = slc
        self.slicetype = type(self.arrslice)

        # cache transformation operations
        self.flattened_bounds = None
        self.bounding_hypercube = None
        self.shape = None
        self.offset = None

    def __repr__(self):
        pass

    @staticmethod
    def flattened_bounds(self, arrslice, db_shape):
        """
        Returns a list of one tuple for each contiguous section of the dbarray described by the ndslice.
        Each tuple contains the index in the dbarray of the first and last elements of the contiguous secion.
        Sections appear in the order they occur in the slice. 

        Args:
            arrslice (ndslice): a slice of a dbarray of shape db_shape
            db_shape (dbarray): shape of a dbarray

        Returns:
            flattened_bounds (list): a list of tuples for each contiguous block of data in arrslice (C-ordered). 
            Each tuple (start, stop) indicates the index of the first and last elements of the contigous block.
        """
        if not arrslice.flattened_bounds is None:
            # cached
            return arrslice.flattened_bounds

        else:
            # needs to compute 
            # number of dimensions in view of dbarray
            ndim = len(db_sape)

            flattened_bounds = []
            if arrslice.slicetype is int:
                # index by a scalar
                if ndim == 1:
                    # slice is single value
                    return [(arrslice, arrslice)]
                elif ndim == 2:
                    # slice is 1D array
                    return [(0, db_shape[1])]
                else:
                    # slice is n-dim array, n > 1
                    sub_shape = db_shape[1:]
                    sub_bounds = []
                    for d in db_shape[1]:
                        new_slice = ndslice(slice(None, None, None))
                        sub_bounds += flattened_bounds(new_slice, subshape)
                    return [(arrslice + start, arrslice + end) for start, end in sub_bounds]

            elif arrslice.slicetype is slice:
                # must parse slice values to scalars

            else:
                # ndslice is a tuple


            arrslice.flattened_bounds = flattened_bounds

            return flattened_bounds

    @staticmethod
    def bounding_hypercube(self, arrslice):
        """
        Calculates indices of minimum hypercube bounding the ndslice arrslice.
        """
        if not self.bounding_hypercube is None:
            # cached
            return self.bounding_hypercube

        else:
            # needs to compute
            bounding_hypercube = 1
            self.bounding_hypercube = bounding_hypercube

            return bounding_hypercube

    @staticmethod
    def shape(self, arrslice):
        """
        Returns dimensions of minimum bounding_hypercube of arrslice.
        """


    @staticmethod
    def offset(self, arrslice):
        """
        Returns index in dbarray object along all dimensions of the [0,...,0] element of the slice.
        """
        if not self.flattened_bounds is None:
            # cached
            return self.flattened_bounds

        else:
            # needs to compute
            offset = 1
            self.offset = offset

            return offset

    @staticmethod
    def ndslice_from_bounds(self, bounds):
        """
        Not sure if we need this.

        Returns ndslice corresponding to linear of hypercube bounds.
        """
        pass

class dbarray(object):
    """
    Implements a ndarray-like object with data stored on disk, not memory. Data size
    is only constrained by available space on disk.
    """
    def __init__(self, shape, dtype, data_dir, max_chunk_size=None):
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
        arrslice = ndslice(slc)
        return self.read(arrslice)

    def __setitem__(self, slc, ndview):
        """
        Overload assignment indexing.
        """
        arrslice = ndslice(slc)
        view = ndview(ndview.asndarray(), ndview.dbarray, arrslice=arrslice)

        self.flush(view)

    def read(self, arrslice=None):
        """
        Returns (in-memory) ndview object corresponding to dbarray[arrslice].
        """
        # read data from disk contained in bounding indices
        flattened_bounds = ndslice.flattened_bounds(arrslice)
        shape = ndslice.shape(arrslice)
        data = self.am.pull(flattened_bounds, shape)
        
        # create ndview
        view = ndview(data, self, arrslice)

        return view

    def flush(self, ndview):
        """
        Pushes data in ndview to dbarray on disk.
        """
        # convert arrslice to flattened bounds
        flattened_bounds = ndslice.flattened_bounds(arrslice, self)

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




