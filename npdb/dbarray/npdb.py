import sys
import operator

import numpy as np

class dbarray(object):
    def __init__(self, shape, dtype, filename, max_slice_size=None):
        # size attributes
        self.shape = tuple(shape)
        self.size = reduce(operator.mul, self.shape)
        self.ndim = len(shape)
        self.max_slice_size = max_slice_size

        # allocate disk space
        try:
            self.filepath = self._alloc()
        except Exception as e:
            print "Disk allocation failed.", e
            sys.exit(e)

    def __len__(self):
        """
        Following NumPy convention, len(dbarray) is the size of the first dimension.
        """
        return self.shape[0]

    def __getitem__(self, indices):
        """
        Overload access indexing.
        """
        return self.slice(indices)

    def __setitem__(self, indices, ndview):
        """
        Overload assignment indexing.
        """
        dims, offset = self._dims_offset_from_indices(indices, ndview)
        view = ndview(ndview.asndarray(), ndview.dbarray, dims=dims, offset=offset)

        self.flush(view)

    def _alloc(self):
        """
        Allocates disk space for dbarray.
        """
        # filepath = D # pointer to file

        # return filepath
        pass

    def _dims_offset_from_indices(self, indices, ndview):
        """
        Returns dims and offset corresponding to indices.
        """
        pass

    def _indices_from_dims_offset(self, indices, ndview):
        """
        Returns dims and offset corresponding to indices.
        """
        pass

    def slice(self, indices=None):
        """
        Returns (in-memory) ndview object corresponding to dbarray[indices].
        """
        # dims = d
        # offset = o
        # data = C # move into memory, create ndarray
        # view = ndview(data, self, dims, offset)

        # return view
        pass

    def flush(self, ndview):
        """
        Pushes data in ndview to dbarray on disk.
        """
        arr = ndview.asndarray()
        np.save("../data/array.dat", arr)

    @classmethod
    def asndarray(self, dbarray, copy):
        """
        Returns view of entire dbarray as ndarray object.
        """
        full_view = self.slice(indices=None)
        return full_view.asndarray(copy)

class ndview(np.ndarray):
    def __new__(cls, data, dbarray, dims=None, offset=None):
        """
        Inherits from np.ndarray; additional parameters locate the view in containing dbarray.
        """
        obj = np.asarray(data).view(cls)
        
        # parent
        obj.dbarray = dbarray

        # location in parent
        obj.dims = dims
        obj.offset = offset
        
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.bounds = getattr(obj, 'bounds', None)

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

    @classmethod
    def merge(self, ndviews, fill=0):
        """
        Uses bounds to merge multiple ndview objects into a single new ndview object.
        """
        # # check that all ndviews have same dtype
        # dtypes = [view.dtype for view in ndviews]
        # assert(all(dtypes[0] == dtype for dtype in dtypes)), "All ndview dtypes must be equal"

        # # calculate bounds and dimensions of merged array
        # full_bounds = [[(b, b + view.shape[dim]) for dim, b in enumerate(view.bounds)] for view in ndviews]
        # flattened_bounds = [[i for s in axis for i in s] for axis in zip(*full_bounds)]
        # merged_bounds = [(min(b), max(b)) for b in flattened_bounds]
        # merged_dim = [u - l for l, u in merged_bounds] 

        # # create new empty ndview object
        # if fill == 0:
        #     merged_arr = np.zeros(shape=merged_dim, dtype=dtypes[0])
        # elif fill == 1:
        #     merged_arr = np.ones(shape=merged_dim, dtype=dtypes[0])
        # else:
        #     merged = np.empty(shape=merged_dim, dtype=dtypes[0])
        #     merged = fill

        # merged = ndview(merged_arr, bounds=merged_bounds)

        # # populate merged ndview with values
        # for view_num, view in enumerate(ndviews):
        #     offset = view.bounds
        #     for idx, val in np.ndenumerate(view):
        #         merged_idx_offset = [sum(i) for i in zip(idx, offset)]
        #         merged_idx = tuple([i - lower[0] for i, lower in zip(merged_idx_offset, merged_bounds)])
        #         merged[merged_idx] = val

        # return merged
        pass


db = dbarray((3,3,3), float, "file")
a = ndview(np.ones(shape=(3,3)), db, offset=(0,0,0))
a.flush()
print a
print type(a)
a.flush()
b = ndview(np.zeros(shape=(3,3)), offset=(0,1,0))
c = ndview([[3,1],[1,5],[2,3]], offset=(1,2))
d = ndview([[3,1],[1,5]], offset=(6,9))


print ndview.merge([a,b])

# c = a.asndarray(copy=True)
# c[0,0] = 100
# print c
# print a
# print type(c)
# print type(a)




