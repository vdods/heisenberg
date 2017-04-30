import itertools
import numpy as np

def __is_nondecreasing_sequence (seq):
    return all(s0 <= s1 for (s0,s1) in zip(seq[:-1],seq[1:]))

def __is_subsequence_of_nondecreasing_sequence (subseq, seq):
    assert __is_nondecreasing_sequence(seq), 'seq is not nondecreasing'

    subseq_index = seq_index = 0
    while True:
        # If we've matched all elements of subseq, then we have a subsequence.
        if subseq_index == len(subseq):
            return True
        # This condition is guaranteed to occur at some point, guaranteeing that the function will return.
        elif seq_index == len(seq):
            return False

        # If the values at the cursors match, then so far so good, increment both cursors.
        if subseq[subseq_index] == seq[seq_index]:
            subseq_index += 1
        # Increment the sequence read cursor regardless.
        seq_index += 1

def __test__is_subsequence_of_nondecreasing_sequence ():
    # Positive tests

    assert __is_subsequence_of_nondecreasing_sequence([], [])
    assert __is_subsequence_of_nondecreasing_sequence([], [0])
    assert __is_subsequence_of_nondecreasing_sequence([], [0,1])
    assert __is_subsequence_of_nondecreasing_sequence([], [0,1,2])

    assert __is_subsequence_of_nondecreasing_sequence([0], [0])
    assert __is_subsequence_of_nondecreasing_sequence([0], [0,1])
    assert __is_subsequence_of_nondecreasing_sequence([0], [0,1,2])

    assert __is_subsequence_of_nondecreasing_sequence([1], [0,1])
    assert __is_subsequence_of_nondecreasing_sequence([1], [0,1,2])

    assert __is_subsequence_of_nondecreasing_sequence([2], [0,1,2])

    assert __is_subsequence_of_nondecreasing_sequence([0,1], [0,1])
    assert __is_subsequence_of_nondecreasing_sequence([0,1], [0,1,2])

    assert __is_subsequence_of_nondecreasing_sequence([0,2], [0,1,2])

    assert __is_subsequence_of_nondecreasing_sequence([1,2], [0,1,2])

    assert __is_subsequence_of_nondecreasing_sequence([0,1,2], [0,1,2])

    # Negative tests

    assert not __is_subsequence_of_nondecreasing_sequence([0], [])
    assert not __is_subsequence_of_nondecreasing_sequence([0], [1])
    assert not __is_subsequence_of_nondecreasing_sequence([1,0], [1])
    assert not __is_subsequence_of_nondecreasing_sequence([0,1,2], [0,1])
    assert not __is_subsequence_of_nondecreasing_sequence([0,1,1], [0,1,2])
    assert not __is_subsequence_of_nondecreasing_sequence([0,1,3], [0,1,2])
    assert not __is_subsequence_of_nondecreasing_sequence([0,4,2], [0,1,2])
    assert not __is_subsequence_of_nondecreasing_sequence([5,1,2], [0,1,2])
    assert not __is_subsequence_of_nondecreasing_sequence([5,6,2], [0,1,2])
    assert not __is_subsequence_of_nondecreasing_sequence([5,6,7], [0,1,2])
    assert not __is_subsequence_of_nondecreasing_sequence([5,6,7], [0,1])
    assert not __is_subsequence_of_nondecreasing_sequence([5,6,7], [0])
    assert not __is_subsequence_of_nondecreasing_sequence([5,6,7], [])

    print('__test__is_subsequence_of_nondecreasing_sequence passed.')

def __compose_index_maps (*index_vv):
    """
    A list is a way to represent a map.  Assume list L has N elements.  Then L can be considered to
    map the elements of the sequence (0,1,...,N-1) to the elements of the sequence (L[0],L[1],...,L[N-1])
    respectively.

    This function composes 
    """

    for domain_index_v,codomain_index_v in zip(reversed(index_vv[1:]),reversed(index_vv[:-1])):
        assert all(0 <= domain_index < len(codomain_index_v) for domain_index in domain_index_v), 'domain error; elements of index list {0} must map into the index range [0,...,{1}) (right endpoint excluded), which index the list {2}'.format(domain_index_v, len(codomain_index_v), codomain_index_v)

    current = np.arange(len(index_vv[-1]))
    for index_v in reversed(index_vv):
        current[:] = [index_v[i] for i in current]
    return current

def __index_map_inverse (index_map_v, index_count):
    """
    Let index_map_inverse_v denote the return value of this function.  It will satisfy:

        np.all(__compose_index_maps(index_map_inverse_v, index_map_v) == np.arange(len(index_map_v)))
    """
    index_map_list_v = list(index_map_v) if type(index_map_v) is not list else index_map_v
    assert len(frozenset(index_map_v)) == len(index_map_v), 'index_map_v (which is {0}) must have nonrepeating elements (i.e. it must be injective as a map)'.format(index_map_v)
    assert all(0 <= i < index_count for i in index_map_v), 'index_map_v (which is {0}) must have elements in the range [0,{1}) (right endpoint excluded)'.format(index_map_v, index_count)
    return np.array([index_map_list_v.index(i) if i in index_map_v else 0 for i in range(index_count)])

def __test__index_map_inverse__case (index_map_v, index_count):
    index_map_inverse_v = __index_map_inverse(index_map_v, index_count)
    # print('')
    # print('index_map_v = {0}, index_count = {1}'.format(index_map_v, index_count))
    # print('index_map_inverse_v = {0}'.format(index_map_inverse_v))
    # print('np.arange(len(index_map_v)) = {0}'.format(np.arange(len(index_map_v))))
    # print('__compose_index_maps(index_map_inverse_v, index_map_v) = {0}'.format(__compose_index_maps(index_map_inverse_v, index_map_v)))
    # print('__compose_index_maps(index_map_v, index_map_inverse_v) = {0}'.format(__compose_index_maps(index_map_v, index_map_inverse_v)))
    assert np.all(__compose_index_maps(index_map_inverse_v, index_map_v) == np.arange(len(index_map_v)))

def __test__index_map_inverse ():
    __test__index_map_inverse__case([], 0)

    __test__index_map_inverse__case([], 1)
    __test__index_map_inverse__case([0], 1)

    __test__index_map_inverse__case([], 2)
    __test__index_map_inverse__case([0], 2)
    __test__index_map_inverse__case([1], 2)
    __test__index_map_inverse__case([0,1], 2)
    __test__index_map_inverse__case([1,0], 2)

    __test__index_map_inverse__case([], 3)
    __test__index_map_inverse__case([0], 3)
    __test__index_map_inverse__case([1], 3)
    __test__index_map_inverse__case([2], 3)
    __test__index_map_inverse__case([0,1], 3)
    __test__index_map_inverse__case([1,0], 3)
    __test__index_map_inverse__case([0,2], 3)
    __test__index_map_inverse__case([2,0], 3)
    __test__index_map_inverse__case([1,2], 3)
    __test__index_map_inverse__case([2,1], 3)
    __test__index_map_inverse__case([0,1,2], 3)
    __test__index_map_inverse__case([0,2,1], 3)
    __test__index_map_inverse__case([1,0,2], 3)
    __test__index_map_inverse__case([1,2,0], 3)
    __test__index_map_inverse__case([2,0,1], 3)
    __test__index_map_inverse__case([2,1,0], 3)

    print('__test__index_map_inverse passed.')

def apply_along_axes (func, input_axis_v, input_array, *args, output_axis_v=None, func_output_shape=None, **kwargs):
    """
    This function is a generalization of numpy.apply_along_axis.

    Let A = (0,1,...,N-1), where N is len(input_array.shape).  input_axis_v should be* a nonempty
    subsequence of A; this means that 0 < len(input_axis_v) <= len(A), each element of input_axis_v
    must be an element of the sequence A, and input_axis_v must be strictly increasing (as A is).

    *Note that input_axis_v may contain values in the range [-N,0), in which case, N is added to
    each negative element to bring them all within the range [0,N) as valid axis indices.  For example,
    this means that -1 addresses the last axis, -2 addresses the second-to-last axis, etc.

    TODO: Allow the base case where input_array.shape = () and input_axis_v = [].

    Let I be the complementary subsequence of input_axis_v, meaning that I is the sequence consisting
    of elements of A not found in input_axis_v, and I is strictly increasing (as A is).

    func is then called on each slice of input_array where the indices specified by I are fixed and
    the indices specified by input_axis_v are free.  The shape of the return value of func on the
    first call is used to determine the shape of the return value of apply_along_axes, or if
    func_output_shape is not None, then func_output_shape is assumed to be the shape of the return
    value of func.

    Let S denote the shape of the return value of func (which may be specified by func_output_shape as
    specified above).  Let B = (0,1,...,M-1), where M is len(I)+len(S).  The return value of
    apply_along_axes will be a numpy.ndarray having M indices.

    If output_axis_v is not None, it must be** a subsequence of B, specifying which axes of the
    return value will be used to index the output of func.  If output_axis_v is None, then it
    will be assumed to have value (L,L+1,...,M-1), where L = len(I), i.e. the output axes will
    be the last axes of the return value.

    **Note that, just as input_axis_v, output_axis_v may contain values in the range [-N,0), in which
    case, N is added to bring each negative element to bring them all within the range [0,N).

    All extra args and kwargs will be passed through to calls to func.

    Note that the single-element input_axis_v special-case call

        apply_along_axes(func, [i], input_array, ...)

    should be equivalent to the standard numpy call

        numpy.apply_along_axis(func, i, input_array, ...)
    """

    assert len(input_axis_v) > 0, 'input_axis_v (which is {0}) must be a nonempty subsequence of (0,1,...,{1})'.format(input_axis_v, len(input_array.shape)-1)

    N = len(input_array.shape)
    A = np.arange(N)
    # Note that the length of the complementary subsequence of N in A is N-len(input_axis_v).

    assert all(-N <= input_axis < N for input_axis in input_axis_v), 'input_axis_v (which is {0}), must contain values in the range of [-N,N) (where N = {1})'.format(input_axis_v, N)

    normalized_input_axis_v = np.fromiter((input_axis if input_axis >= 0 else input_axis+N for input_axis in input_axis_v), np.int, len(input_axis_v))

    # print('input_array.shape = {0}'.format(input_array.shape))

    assert __is_subsequence_of_nondecreasing_sequence(normalized_input_axis_v, A), 'normalized_input_axis_v (which is {0}) must be a nonempty subsequence of [0,{1}) (right endpoint excluded)'.format(normalized_input_axis_v, len(input_array.shape))

    is_not_normalized_input_axis_v = np.array([axis_index not in normalized_input_axis_v for axis_index in A])
    non_normalized_input_axis_v = np.array([axis_index for axis_index in A if is_not_normalized_input_axis_v[axis_index]])
    input_iterator_v = [range(input_array.shape[axis_index]) if is_not_normalized_input_axis_v[axis_index] else [slice(None)] for axis_index in A]
    # print('is_not_normalized_input_axis_v = {0}'.format(is_not_normalized_input_axis_v))
    # print('non_normalized_input_axis_v = {0}'.format(non_normalized_input_axis_v))
    # print('input_iterator_v = {0}'.format(input_iterator_v))

    # If func_output_shape is not specified, then derive it.
    if func_output_shape is None:
        func_output_shape = np.shape(func(input_array[tuple(0 if axis_index not in normalized_input_axis_v else slice(None) for axis_index in A)], *args, **kwargs))
    # print('func_output_shape = {0}'.format(func_output_shape))

    # B = np.arange(N-len(normalized_input_axis_v)+len(func_output_shape))
    B = np.arange(len(non_normalized_input_axis_v)+len(func_output_shape))
    M = len(B)
    # print('B = {0}'.format(B))

    # If output_axis_v is None, then it will be assumed to be the trailing axes of the output.
    if output_axis_v is None:
        output_axis_v = np.arange(-len(func_output_shape), 0)
    # print('normalized_input_axis_v = {0}'.format(normalized_input_axis_v))
    # print('output_axis_v = {0}'.format(output_axis_v))

    assert len(func_output_shape) == len(output_axis_v), 'func_output_shape (which is {0}) must have same number of elements as output_axis_v (which is {1})'.format(func_output_shape, output_axis_v)

    normalized_output_axis_v = np.fromiter((output_axis if output_axis >= 0 else output_axis+M for output_axis in output_axis_v), np.int, len(output_axis_v))

    assert __is_subsequence_of_nondecreasing_sequence(normalized_output_axis_v, B), 'normalized_output_axis_v (which is {0}) must be a subsequence of [0,{1}) (right endpoint excluded)'.format(normalized_output_axis_v, len(B))

    is_not_output_axis_v = [axis_index not in normalized_output_axis_v for axis_index in B]
    # print('is_not_output_axis_v = {0}'.format(is_not_output_axis_v))
    non_output_axis_v = [axis_index for axis_index in B if is_not_output_axis_v[axis_index]]
    # print('non_output_axis_v = {0}'.format(non_output_axis_v))
    non_output_axis_inv_v = __index_map_inverse(non_output_axis_v, len(B))
    # print('non_output_axis_inv_v = {0}'.format(non_output_axis_inv_v))
    output_axis_inv_v = __index_map_inverse(normalized_output_axis_v, len(B))
    # print('output_axis_inv_v = {0}'.format(output_axis_inv_v))
    # print('__compose_index_maps(non_normalized_input_axis_v, non_output_axis_inv_v) = {0}'.format(__compose_index_maps(non_normalized_input_axis_v, non_output_axis_inv_v)))
    output_iterator_v = [range(input_array.shape[non_normalized_input_axis_v[non_output_axis_inv_v[axis_index]]]) if is_not_output_axis_v[axis_index] else [slice(None)] for axis_index in B]
    # print('output_iterator_v = {0}'.format(output_iterator_v))

    retval = np.ndarray(tuple(input_array.shape[non_normalized_input_axis_v[non_output_axis_inv_v[axis_index]]] if is_not_output_axis_v[axis_index] else func_output_shape[output_axis_inv_v[axis_index]] for axis_index in B), dtype=input_array.dtype)
    # print('retval.shape = {0}'.format(retval.shape))
    for input_I,output_I in zip(itertools.product(*input_iterator_v), itertools.product(*output_iterator_v)):
        # print('input_I = {0}, output_I = {1}'.format(input_I, output_I))
        retval[output_I] = func(input_array[input_I], *args, **kwargs)

    # print('')
    return retval

def __test__apply_along_axes__compare_with__apply_across_axis ():
    rng = np.random.RandomState(42)
    a = rng.randn(4,5,6,7)
    assert np.all(apply_along_axes(np.sum, [0], a) == np.apply_along_axis(np.sum, 0, a))
    assert np.all(apply_along_axes(np.sum, [1], a) == np.apply_along_axis(np.sum, 1, a))
    assert np.all(apply_along_axes(np.sum, [2], a) == np.apply_along_axis(np.sum, 2, a))
    assert np.all(apply_along_axes(np.sum, [3], a) == np.apply_along_axis(np.sum, 3, a))
    print('__test__apply_along_axes__compare_with__apply_across_axis passed.')

def __test__apply_along_axes ():
    def symmetric_square (m):
        return np.einsum('ij,kj->ik', m, m)

    def is_symmetric (m):
        return np.all(m == m.T)

    rng = np.random.RandomState(42)
    a = rng.randn(2,3,4,5)
    N = len(a.shape)

    # Use all possible combinations of input and output axes.
    for input_i0 in range(-N,N-1):
        for input_i1 in range(input_i0+1,N):
            # Only test the pairs where input_i0 indexes an axis before that indexed by input_i1.
            if input_i0 + N >= input_i1:
                continue

            for output_i0 in range(-N,N-1):
                for output_i1 in range(output_i0+1,N):
                    # Only test the pairs where output_i0 indexes an axis before that indexed by output_i1.
                    if output_i0 + N >= output_i1:
                        continue

                    output_axis_v = (output_i0,output_i1)
                    # Compute the result.  The multi-slice across the output axes should be a symmetric matrix.
                    result = apply_along_axes(symmetric_square, (input_i0,input_i1), a, output_axis_v=output_axis_v)
                    # Figure out which indices correspond to the input axes; call these result_non_output_axis_v.
                    all_indices = tuple(range(N))
                    normalized_output_axis_v = tuple(output_axis if output_axis >= 0 else output_axis+N for output_axis in output_axis_v)
                    result_non_output_axis_v = sorted(list(frozenset(all_indices) - frozenset(normalized_output_axis_v)))
                    # print('output_axis_v = {0}, result_non_output_axis_v = {1}'.format(output_axis_v, result_non_output_axis_v))
                    assert len(result_non_output_axis_v) == 2
                    # Take all multi-slices and verify symmetry.
                    for check_i0,check_i1 in itertools.product(range(result.shape[result_non_output_axis_v[0]]), range(result.shape[result_non_output_axis_v[1]])):
                        # Construct the multi-slice to take.
                        multislice = [slice(None) for _ in range(4)]
                        multislice[result_non_output_axis_v[0]] = check_i0
                        multislice[result_non_output_axis_v[1]] = check_i1
                        # print('multislice = {0}'.format(multislice))
                        assert is_symmetric(result[multislice])

    print('__test__apply_along_axes passed.')

if __name__ == '__main__':
    __test__is_subsequence_of_nondecreasing_sequence()
    __test__index_map_inverse()
    __test__apply_along_axes__compare_with__apply_across_axis()
    __test__apply_along_axes()
