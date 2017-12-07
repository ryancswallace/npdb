

def merge(dbviews, fill=0):
        """
        Uses dbview positions to merge multiple dbview objects into a single new dbview object.

        TODO: fix merged_dims to account for parallel slices. Find bounding hypercube.
        """
        # parent_ndim = dbviews[0].dbarray.ndim

        # # check that all dbviews have same dtype
        # dtypes = [view.dtype for view in dbviews]
        # assert(all(dtypes[0] == dtype for dtype in dtypes)), "All dbview dtypes must be equal"

        # # calculate dims and offsets of merged array
        # merged_dims = list(set(sorted(itertools.chain.from_iterable([view.dims for view in dbviews]))))
        # offsets_by_dim = [[view.offset[dim] for view in dbviews] for dim in range(parent_ndim)]
        # offsets_by_view = [[offsets[view_idx] for offsets in offsets_by_dim] for view_idx in range(len(dbviews))]
        # merged_offset = [min(offsets) for offsets in offsets_by_dim]

        # # calculate position of merged view in dbarray
        # length_by_view = [[view.shape[view.dims.index(dim)] if dim in view.dims else 1 for view in dbviews] for dim in range(parent_ndim)]
        # upper_bounds = [max([view.offset[dim] + length_by_view[dim][view_idx] for view_idx, view in enumerate(dbviews)]) for dim in range(parent_ndim)]
        # merged_bounds = zip(merged_offset, upper_bounds)
        # length_by_dim =  [u - l for l, u in merged_bounds] 
        # length_by_merged_dim = [l for l in length_by_dim if l != 1]
        # flattened = [l == 1 for l in length_by_dim]
        # flattened_dims = list(itertools.compress(range(parent_ndim), flattened))

        # # create new empty dbview object
        # if fill == 0:
        #     merged_arr = np.zeros(shape=length_by_merged_dim, dtype=dtypes[0])
        # elif fill == 1:
        #     merged_arr = np.ones(shape=length_by_merged_dim, dtype=dtypes[0])
        # else:
        #     merged_arr = np.empty(shape=length_by_merged_dim, dtype=dtypes[0])
        #     merged_arr[:] = fill

        # merged = dbview(merged_arr, dbviews[0].dbarray, dims=merged_dims, offset=merged_offset)
        
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

        # for view_idx, view in enumerate(dbviews): 
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
        # hypercube = _bounding_hypercube(dbviews)

        # # find offset of merged array relative to containing dbarray
        # merged_offset = 