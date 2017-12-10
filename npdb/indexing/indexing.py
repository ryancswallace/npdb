"""
Implements indexing on a dbarray. dbarray indexing maps a raw bracket index to
a set of maximal C-contiguous simple index bounds containing the indexed data.

The dbindx an index of a dbarray. A dbindex is created from either a bracketed 
expression, i.e., dbarray[raw_idx], or from explicit dbindex creation functions.
"""

# TODO: indexed_shape
# TODO: ellipsis, None, newaxis, fancy

import numpy as np

def unravel_index(db_shape, raw_idx):
    """
    Converts a raw bracked index expression to the corresponding set of dbarray
    index bounds and the shape of the resulting indexed array; i.e., performs 
    dbarray[raw_idx] where dbarray has shape db_shape.

    Supported raw_idx types are int, slice, tuple, np.ellipsis, np.newaxis, int 
    array, boolean array, and array-like.

    Indexing type is determined by the value of raw_idx and corresponds to one 
    of the numpy indexing types: basic indexing or advanced (fancy) indexing.
    Numpy field access (and structured arrays generally) is not supported.

    Args:
        db_shape (tuple): shape of bdarray being indexed.
        raw_idx (one of supported raw index types): indices to retrieve or set

    Returns:
        index_bounds (list): a list of tuples of index bounds for each maximal
        C-contiguous block of data in dbarray[raw_idx]. Each tuple (start, stop)
        indicates the index of the first and last elements of the contigous 
        block.
        indexed_shape (tuple): the shape of dbarray[raw_idx]
    """    
    # first format index and determine the type of index
    idx, idx_type = to_dbindex(raw_idx) 

    if idx_type == "basic":
        # basic indexing
        index_bounds = simple_index(db_shape, idx)
    else:
        # fancy indexing
        indexed_shape, index_bounds = fancy_index(db_shape, idx)

    maximal_index_bounds = merge_contiguous(index_bounds)

    # TODO: replace
    return (3,3,3), maximal_index_bounds
    
def to_dbindex(raw_idx):
    """
    Given a raw index, returns the formatted index and the type of indexing. 
    Also checks that raw index is valid.

    Args:
        raw_idx (one of dbindex_types): a raw index of a dbarray.

    Returns:
        idx, idx_type (tuple): a formatted index value and the type index
    """
    # a dbindex can be one of the following types 
    dbindex_types = [int, slice, tuple, type(Ellipsis), type(None), np.ndarray]

    # coerce array-likes into np.ndarrays and check that arrays are of bools 
    # or ints 
    if not any([isinstance(raw_idx, t) for t in dbindex_types]):
        # try to cast np.ndarray
        try:
            raw_idx = np.array(raw_idx)
        except Exception:
            raise TypeError, "dbarray index of unsupported type"

    if isinstance(raw_idx, np.ndarray):
        # array index must be of integers or bools
        assert(raw_idx.dtype in [int, bool])

    array_in_tuple = False
    if isinstance(raw_idx, tuple):
        raw_idx_list = list(raw_idx)
        # any tuple elements that are array-like must be of ints or bools
        for e_idx, e in enumerate(raw_idx):
            if not any([isinstance(e, t) for t in dbindex_types]):
                # either array-like
                try:
                    raw_idx_list[e_idx] = np.array(e)
                    array_in_tuple = True
                except Exception:
                    # or invalid
                    raise TypeError, "dbarray index of unsupported type"
        raw_idx = tuple(raw_idx_list)

    idx = raw_idx

    # determine indexing type
    idx_type = "fancy" if isinstance(idx, np.ndarray) or array_in_tuple else "basic"
    assert(any([isinstance(idx, t) for t in dbindex_types])), "Error in index formatting"
 
    return idx, idx_type

def merge_contiguous(index_bounds):
    """
    Combines any contiguous index bounds.

    Args:
        index_bounds (list): a list of bounding index tuples for each 
        potentially non-maximal C-contiguous block of data selected.

    Returns:
        maximal_index_bounds (list): a list of bounds equivalent to
        index_bounds, but such that each bounded area is maximal, i.e., no
        pairs of consecutive bounds can be merged to form a single bounds. 
    """
    # combine any contiguous index bounds
    maximal_index_bounds = []

    if len(index_bounds) < 2:
        # nothing to merge
        return index_bounds
    else:
        # for the first element compare the consecutive bounds in index_bounds
        if index_bounds[0][1] != index_bounds[1][0]:
            # not contiguous
            maximal_index_bounds += [index_bounds[0], index_bounds[1]]
        else:
            # contiguous
            maximal_index_bounds.append((index_bounds[0][0], 
                                         index_bounds[1][1]))
        for i in range(1, len(index_bounds) - 1):
            # compare to last element of maximal_index_bounds
            if maximal_index_bounds[-1][1] != index_bounds[i+1][0]:
                # not contiguous
                maximal_index_bounds.append(index_bounds[i+1])
            else:
                # contiguous
                last_bounds = maximal_index_bounds.pop()
                maximal_index_bounds.append((last_bounds[0], 
                                             index_bounds[i+1][1]))

    return maximal_index_bounds

def positivize_idx(axis_len, idx_1d):
    """
    Standardizes a 1d index by converting a negative scalar to a
    corresponding positive scalar. Also checks bounds.
    """ 
    if isinstance(idx_1d, tuple):
        # numerical index
        if idx_1d < 0:
            # convert to positive index
            standardized_idx = idx_1d + axis_len
        else:
            standardized_idx = idx_1d        
        if standardized_idx < 0 or standardized_idx > axis_len - 1:
            # out of bounds
            raise IndexError, ("Index {} out of bounds for axis with length {}"
                               ).format(idx_1d, axis_len)
    else:
        # None, etc.
        standardized_idx = idx_1d

    return standardized_idx

def simple_index(db_shape, idx):
    """
    Implements numpy basic slicing.

    Returns a list of one tuple for each C-contiguous section of the dbarray 
    described by the index. Each tuple contains the index in the dbarray of the 
    first and last elements of the contiguous secion. Sections appear in the 
    order they occur in the resulting indexed array. 

    Args:
        db_shape (tuple): shape of a dbarray to be indexed
        idx (one of supported_index_types): an index of a dbarray

    Returns:
        indexed_shape (tuple): the shape of dbarray[raw_idx]
        index_bounds (list): a list of bounding index tuples for each 
        C-contiguous block of data in the indexed result. 
    """
    # number of dimensions to slice in
    ndim = len(db_shape)

    index_bounds = []
    if isinstance(idx, int):
        # indexing by a scalar
        idx = positivize_idx(db_shape[0], idx)
        if ndim == 1:
            # result is single value
            return [((idx,), (idx,))]
        elif ndim == 2:
            # slice is 1D array
            return [((idx, 0), (idx, db_shape[1] - 1))]
        else:
            # slice is n-dim array, n > 1; recurse down
            sub_idx = to_dbindex(slice(None))[0]
            sub_shape = db_shape[1:]
            sub_bounds = simple_index(sub_shape, sub_idx)
            return [((idx,) + start_idx, (idx,) + end_idx) for
                    start_idx, end_idx in sub_bounds]

    elif isinstance(idx, slice):
        # parse slice values to scalars
        if idx == slice(None):
            # select all
            last_idx = tuple((i - 1 for i in db_shape))
            return [((0,) * ndim, last_idx)]
        else:
            if ndim == 1:
                start = idx.start if not idx.start is None else 0
                stop = idx.stop if not idx.stop is None else db_shape[0]
                if idx.step is None:
                    # simple slice of 1D array
                    start = positivize_idx(db_shape[0], start)
                    stop = positivize_idx(db_shape[0], stop - 1)
                    return [((start,), (stop,))]
                else:
                    # subselection along first axis; slice non-contiguous
                    step = idx.step if not idx.step is None else 1
                    first_axis_idxs = range(start, stop, step)
                    index_bounds = []
                    for first_idx in first_axis_idxs:
                        index_bounds += [((first_idx,), (first_idx,))]
                    return index_bounds
            else:
                # slice along first axis, full slice from higher axes
                sub_idx = to_dbindex(slice(None))[0]
                sub_shape = db_shape[1:]
                sub_bounds = simple_index(sub_shape, sub_idx)
                if idx.step is None:
                    # simple slice of 1D array
                    first_axis_start = idx.start if not idx.start is None else 0
                    first_axis_stop = (idx.stop - 1 if not idx.stop is None else
                                       db_shape[0] - 1)
                    first_axis_start = positivize_idx(db_shape[0],
                                                      first_axis_start)
                    first_axis_stop = positivize_idx(db_shape[0], 
                                                     first_axis_stop)
                    return [((first_axis_start,) + start_idx, 
                            (first_axis_stop,) + end_idx) for start_idx, 
                            end_idx in sub_bounds]
                else:
                    # subselection along first axis
                    index_bounds = []
                    start = idx.start if not idx.start is None else 0
                    stop = idx.stop if not idx.stop is None else db_shape[0]
                    step = idx.step if not idx.step is None else 1
                    start = positivize_idx(db_shape[0], start)
                    stop = positivize_idx(db_shape[0], stop)
                    first_axis_idxs = range(start, stop, step)
                    for first_idx in first_axis_idxs:
                        index_bounds += [((first_idx,) + start_idx, (first_idx,)
                                         + end_idx) for start_idx, end_idx in 
                                         sub_bounds]
                    return index_bounds

    else:
        # indexing by a tuple
        if len(idx) > ndim:
            # too many indices; out of bounds
            raise IndexError, (("Too many indices for array with dimension {}")
                               .format(ndim))
        # get bounds on first axis
        first_axis_idxs = simple_index((db_shape[0],), to_dbindex(idx[0])[0])
        if ndim == 1:
            # no subaxes
            return first_axis_idxs
        else:
            if len(idx) > 1:
                # next get bounds on subaxes
                sub_idx = idx[1:]
            else:
                # no bounds on subaxes
                sub_idx = slice(None)
            sub_shape = db_shape[1:]
            sub_bounds = simple_index(sub_shape, to_dbindex(sub_idx)[0])
            # combine first axis bounds with subaxis bounds
            index_bounds = []
            for first_axis_start, first_axis_stop in first_axis_idxs:
                index_bounds += [(first_axis_start + sub_axis_start, 
                           first_axis_stop + sub_axis_stop) for sub_axis_start,
                           sub_axis_stop in sub_bounds]
            return index_bounds

def fancy_index(db_shape, idx):
    """
    Implements numpy fancy slicing.

    Returns a list of one tuple for each C-contiguous section of the dbarray 
    described by the index. Each tuple contains the index in the dbarray of the 
    first and last elements of the contiguous secion. Sections appear in the 
    order they occur in the resulting indexed array. 

    Arguments and return values same as those of basic_index().
    """
    pass

def bounding_hypercube(self, arrslice):
    """
    Calculates indices of minimum hypercube bounding the dbindex arrslice.
    """
    if not self.bounding_hypercube is None:
        # cached
        return self.bounding_hypercube

    else:
        # needs to compute
        bounding_hypercube = 1
        self.bounding_hypercube = bounding_hypercube

        return bounding_hypercube

def shape(self, arrslice):
    """
    Returns dimensions of minimum bounding_hypercube of arrslice.
    """
    pass

def offset(self, arrslice):
    """
    Returns index in dbarray object along all dimensions of the [0,...,0] element of the slice.
    """
    if not self.bounds is None:
        # cached
        return self.bounds

    else:
        # needs to compute
        offset = 1
        self.offset = offset

        return offset

def dbindex_from_bounds(self, bounds):
    """
    Not sure if we need this.

    Returns dbindex corresponding to linear of hypercube bounds.
    """
    pass
