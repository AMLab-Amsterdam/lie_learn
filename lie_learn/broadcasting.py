
import numpy as np
from numpy.lib.index_tricks import as_strided


def generalized_broadcast(arrays):
    """
    Broadcast X and Y, while ignoring the last axis of X and Y.

    If X.shape = xs + (i,)
    and Y.shape = ys + (j,)
    then the output arrays have shapes
    Xb.shape = zs + (i,)
    Yb.shape = zs + (j,)
    where zs is the shape of the broadcasting of xs and ys shaped arrays.

    :param arrays: a list of numpy arrays to be broadcasted while ignoring the last axis.
    :return: a list of arrays whose shapes have been broadcast
    """
    arrays1 = np.broadcast_arrays(*[A[..., 0] for A in arrays])
    shapes_b = [A1.shape + (A.shape[-1],) for A1, A in zip(arrays1, arrays)]
    strides_b = [A1.strides + (A.strides[-1],) for A1, A in zip(arrays1, arrays)]
    arrays_b = [as_strided(A, shape=shape_Ab, strides=strides_Ab)
                for A, shape_Ab, strides_Ab in zip(arrays, shapes_b, strides_b)]
    return arrays_b


def make_gufunc(f, core_dims_in, core_dims_out):
    """
    Automatically turn a function f into a generalized universal function (gufunc).

    :param f:
    :param core_dims_in:
    :param core_dims_out:
    :return:
    """

    return

    def gufunc(args):
        args = generalized_broadcast(args)
        data_shape = args[0].shape[:-len(core_dims_in[0])]
        args = [A.reshape(-1, A.shape[-1]) for A in args]

        #if X_out is None:
        #    X_out = np.empty_like(X)
        #X_out = X_out.reshape(-1, X.shape[-1])

        out = f(args)

        return out.reshape()

    return gufunc
