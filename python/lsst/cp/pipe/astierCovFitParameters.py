# I don't know how wrote that. It belonged to "saunerie", developed
# at LPNHE (Paris). Just copied here.

"""

This module contains utility classes to handle parameters in linear and
non-linear least square fits implemented in linearmodels and
nonlinearmodels. It provide 2 main features:

- `structarray`: a derivative of numpy class ndarray to manage large
vectors organized into named subgroups.


- `FitParameters` : a class to manage large parameter vectors.  It
allows to easily fix/release specific parameters or entire subgroups,
and remap the remaining free parameters into a contiguous vector.
"""

import numpy as np

class Structure(object):
    """ Collection of named slices

    Slices are specified by a name and a length. If omitted, the length
    default to one and the name to __0__ for the first unamed slice, __1__
    for the second and so on.

    Examples:
    ---------
    >>> s = Structure([('a', 7), 3])
    >>> print len(s)
    10
    >>> for name in s: print name
    a
    __0__
    >>> print len(Structure([('a', 3), 'b']))
    4
    """
    def __init__(self, groups):
        if isinstance(groups, Structure):
            self.slices = groups.slices.copy()
            self._len = groups._len
        else:
            self.slices = {}
            i = 0
            _n_unnamed = 0
            for group in groups:
                if isinstance(group, int):
                    name = "__%d__" % _n_unnamed
                    _len = group
                    _n_unnamed += 1
                elif isinstance(group, str):
                    name = group
                    _len = 1
                else:
                    try:
                        name, _len = group
                    except TypeError:
                        raise TypeError('Structure specification not understood: %s' % repr(group))
                self.slices[name] = slice(i, i + _len)
                i += _len
            self._len = i
        
    def __getitem__(self, arg):
        if isinstance(arg, str):
            return self.slices[arg]
        else:
            return arg
        
    def __iter__(self):
        return self.slices.__iter__()

    def __len__(self):
        return self._len
    
class structarray(np.ndarray):
    """Decorate numpy arrays with a collection of named slices.

    Array slices becomes accessible by their name. This is applicable to
    nd array, although the same `Structure` is shared between all
    dimensions.
    
    Examples:
    ---------
    >>> v = structarray(np.zeros(10), [('a', 3), ('b', 7)])
    >>> print v['a']
    [ 0.  0.  0.]

    >>> C = structarray(np.zeros((10,10)), [('a', 2), ('b', 8)])
    >>> print C['a', 'a']
    [[ 0.  0.]
     [ 0.  0.]]
    """
    def __new__(cls, array, struct=[]):
        obj = array.view(cls)
        obj.struct = Structure(struct)
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.struct = getattr(obj, 'struct', [])

    def __getitem__(self, args):
        if not isinstance(args, tuple):
            args = args,
        newargs = tuple([self.struct[arg] for arg in args])
        return np.asarray(self)[newargs]

    def __setitem__(self, args, val):
        if not isinstance(args, tuple):
            args = args,
        newargs = tuple([self.struct[arg] for arg in args])
        np.asarray(self)[newargs] = val

    def __getslice__(self, start, stop):
        return self.__getitem__(slice(start, stop))

    def __reduce__(self):
        # Get the parent's __reduce__ tuple
        pickled_state = super(structarray, self).__reduce__()
        # Create our own tuple to pass to __setstate__
        new_state = pickled_state[2] + (self.struct,)
        # Return a tuple that replaces the parent's __setstate__ tuple with our own
        return (pickled_state[0], pickled_state[1], new_state)

    def __setstate__(self, state):
        self.struct = state[-1]  # Set the info attribute
        # Call the parent's __setstate__ with the other tuple elements.
        super(structarray, self).__setstate__(state[0:-1])
        
  
class FitParameters(object):
    """ Manages a vector of fit parameters with the possibility to mark a subset of
    them as fixed to a given value.

    The parameters can be organized in named slices (block of contiguous
    values) accessible through indexing by their name as in `structarray`.
    
    >>> p_salt = FitParameters(['X0', 'X1', 'Color', 'Redshift'])
    >>> p_dice = FitParameters([('alpha', 2), ('S', 10), ('dSdT', 10), 'idark'])
        
    It is possible to modify the parameters in place. Using the indexing
    of slices by name simplifies somewhat the operations, as one does
    not need to care about the position of a slice within the entire
    parameter vector:

    >>> p_dice['idark'][0] = -1.0232E-12 
    >>> p_dice['S'][4] = 25.242343E-9
    >>> p_dice['dSdT'][:] = 42.
    >>> print p_dice
    alpha: array([ 0.,  0.])
    S: array([  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
             0.00000000e+00,   2.52423430e-08,   0.00000000e+00,
             0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
             0.00000000e+00])
    dSdT: array([ 42.,  42.,  42.,  42.,  42.,  42.,  42.,  42.,  42.,  42.])
    idark: array([ -1.02320000e-12])

    It is also possible to mark parameters as fixed to a value.
    >>> p_dice.fix(0, 12.)
    
    Value is optional. The above is equivalent to:
    >>> p_dice[0] = 12.
    >>> p_dice.fix(0)

    Again named slices simplifies the operations:
    >>> p_dice['S'].fix([0, -1], 12.)
    >>> p_dice['dSdT'].fix([0, -1])
    
    One can fix entire slices at once: 
    >>> p_dice['idark'].fix()
    >>> p_salt['Redshift'].fix(val=0.23432)
    
    The property ``full'' give access to the vector of parameters. The
    property "free" gives access to the free parameters:
    >>> print len(p_dice.free), len(p_dice.full)
    17 23
    
    Note that free relies on fancy indexing. Access thus trigger a
    copy. As a consequence, the following will not actually alter the
    data:
    >>> p_dice.free[0] = 12.
    >>> print p_dice.free[0]
    0.0

    It is still possible to set slices of free parameters as a
    contiguous vector. For example:
    >>> p_dice['S'].free = 12.
    >>> print p_dice['S'].free
    [ 12.  12.  12.  12.  12.  12.  12.  12.]

    >>> p_dice[:5].free = 4.
    >>> print p_dice[:5].free
    [ 4.  4.  4.]
    
    In particular, the typical use case which consists in updating the
    free parameters with the results of a fit works as expected:
    >>> p = np.arange(len(p_dice.free))
    >>> p_dice.free = p

    Last the class provide a convenience function that return the index
    of a subset of parameters in the global free parameters vector, and
    -1 for fixed parameters:
    >>> print p_dice['dSdT'].indexof()
    [-1  9 10 11 12 13 14 15 16 -1]
    >>> print p_dice['dSdT'].indexof([1,2])
    [ 9 10]
    """
    def __init__(self, groups):
        struct = Structure(groups)
        self._free = structarray(np.ones(len(struct), dtype='bool'), struct)
        self._pars = structarray(np.zeros(len(struct), dtype='float'), struct)
        self._index = structarray(np.zeros(len(struct), dtype='int'), struct)
        self._struct = struct
        self._reindex()
        self._base = self
        
    def copy(self):
        cop = FitParameters(self._struct)
        cop._free = structarray(np.copy(self._free), cop._struct)
        cop._pars = structarray(np.copy(self._pars), cop._struct)
        cop._index= structarray(np.copy(self._index), cop._struct)
        cop._reindex()
        cop._base = cop
        return cop
    
    def _reindex(self):
        self._index[self._free] = np.arange(self._free.sum())
        self._index[~self._free] = -1
        
    def fix(self, keys=slice(None), val=None):
        self._free[keys] = False
        if val is not None:
            self._pars[keys] = val
        self._base._reindex()
        
    def release(self, keys=slice(None)):
        self._free[keys] = True
        self._base._reindex()
        
    def indexof(self, indices=slice(None)):
        return self._index[indices]
    
    def __getitem__(self, args):
        # Prevent conversion to scalars
        if isinstance(args, int):
            args = slice(args, args + 1)
        new = FitParameters.__new__(FitParameters)
        new._free = self._free[args]
        new._pars = self._pars[args]
        new._index = self._index[args]
        new._base = self._base
        return new

    def __setitem__(self, args, val):
        self._pars[args] = val
        
    @property
    def free(self):
        return self._pars[self._free]

    @free.setter
    def free(self, val):
        self._pars[self._free] = val

    @property
    def full(self):
        return self._pars

    @full.setter
    def full(self, val):
        self._pars = val

    def __repr__(self):
        if hasattr(self, '_struct'):
            s = "\n".join(['%s: %s' % (key, repr(self._pars[key])) for key in self._struct])
        else:
            s = repr(self._pars)
        return s
    
if __name__ == "__main__":
    s = [('a', 3), ('b', 7)]
    a = structarray(np.random.randn(10), s)
    C = structarray(np.random.randn(10, 10), s)
    p = FitParameters(s)

    import cPickle as pickle
    with open('toto.pkl', 'w') as fid:
        pickle.dump(p, fid)
    with open('toto.pkl') as fid:
        t = pickle.load(fid)
