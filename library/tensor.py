import itertools
import numpy as np
import operator

def tensor_order (T):
    return len(T.shape) if hasattr(T,'shape') else 0

def tensor_shape (T):
    return T.shape if hasattr(T,'shape') else tuple()

def tensor_component (T, multiindex):
    return T[multiindex] if hasattr(T,'shape') else T

def multiindex_iterator (shape):
    return itertools.product(*tuple(range(dim) for dim in shape))

def contract (contraction_string, *tensors, **kwargs):
    def positions_of_all_occurrences_of_char (s, c):
        for pos,ch in enumerate(s):
            if ch == c:
                yield pos
    
    output_index_string = kwargs.get('output', None)
    assert 'dtype' in kwargs, 'Must specify the \'dtype\' keyword argument (e.g. dtype=float, dtype=object, etc).'
    dtype = kwargs['dtype']
    error_messages = []
    
    #
    # Starting here is just checking that the contraction is well-defined, such as checking
    # the summation semantics of the contracted and free indices, checking that the contracted
    # slots' dimensions match, etc.
    # 
    
    # Verify that the indices in the contraction string match the orders of the tensor arguments.
    index_strings = contraction_string.split(',')
    assert len(index_strings) == len(tensors), 'There must be the same number of comma-delimited index strings (which in this case is {0}) as tensor arguments (which in this case is {1}).'.format(len(index_strings), len(tensors))
    all_index_counts_matched = True
    for i,(index_string,tensor) in enumerate(itertools.izip(index_strings,tensors)):
        if len(index_string) != tensor_order(tensor):
            error_messages.append('the number of indices in {0}th index string \'{1}\' (which in this case is {2}) did not match the order of the corresponding tensor argument (which in this case is {3})'.format(i, index_string, len(index_string), tensor_order(tensor)))
            all_index_counts_matched = False
    assert all_index_counts_matched, 'At least one index string had a number of indices that did not match the order of its corresponding tensor argument.  In particular, {0}.'.format(', '.join(error_messages))

    # Determine which indices are to be contracted (defined as any indices occurring more than once)
    # and determine the free indices (defined as any indices occurring exactly once).
    indices = frozenset(c for c in contraction_string if c != ',')
    contraction_indices = frozenset(c for c in indices if contraction_string.count(c) > 1)
    free_indices = indices - contraction_indices # Set subtraction    
    
    # If the 'output' keyword argument wasn't specified, use the alphabetization of free_indices
    # as the output indices.
    if output_index_string == None:
        output_indices = free_indices
        output_index_string = ''.join(sorted(list(free_indices)))
    # Otherwise, perform some verification on output_index_string.
    else:
        # If the 'output' keyword argument was specified (stored in output_index_string), 
        # then verify that it's well-defined, in that that output_index_string contains
        # unique characters.
        output_indices = frozenset(output_index_string)
        output_indices_are_unique = True
        for index in output_indices:
            if output_index_string.count(index) > 1:
                error_messages.append('index \'{0}\' occurs more than once'.format(index))
                output_indices_are_unique = False
        assert output_indices_are_unique, 'The characters of the output keyword argument (which in this case is \'{0}\') must be unique.  In particular, {1}.'.format(output_index_string, ', '.join(error_messages))
        # Verify that free_indices and output_index_string contain exactly the same characters.
        assert output_indices == free_indices, 'The output indices (which in this case are \'{0}\') must be precisely the free indices (which in this case are \'{1}\').'.format(''.join(sorted(output_indices)), ''.join(sorted(free_indices)))

    # Verify that the dimensions of each of contraction_indices match, while constructing
    # an indexed list of the dimensions of the contracted slots.
    contraction_index_string = ''.join(sorted(list(contraction_indices)))
    contracted_indices_dimensions_match = True
    for contraction_index in contraction_index_string:
        indexed_slots_and_dims = []
        for arg_index,(index_string,tensor) in enumerate(itertools.izip(index_strings,tensors)):
            for slot_index in positions_of_all_occurrences_of_char(index_string,contraction_index):
                indexed_slots_and_dims.append((arg_index,slot_index,tensor.shape[slot_index]))
        distinct_dims = frozenset(dim for arg_index,slot_index,dim in indexed_slots_and_dims)
        if len(distinct_dims) > 1:
            slot_indices = ','.join('{0}th'.format(slot_index) for _,slot_index,_ in indexed_slots_and_dims)
            arg_indices = ','.join('{0}th'.format(arg_index) for arg_index,_,_ in indexed_slots_and_dims)
            dims = ','.join('{0}'.format(dim) for _,_,dim in indexed_slots_and_dims)
            error_messages.append('index \'{0}\' is used to contract the {1} slots respectively of the {2} tensor arguments whose respective slots have non-matching dimensions {3}'.format(contraction_index, slot_indices, arg_indices, dims))
            contracted_indices_dimensions_match = False
    assert contracted_indices_dimensions_match, 'The dimensions of at least one set of contracted tensor slots did not match.  In particular, {0}.'.format(', '.join(error_messages))

    def dims_of_index_string (index_string):
        def tensor_and_slot_in_which_index_occurs (index):
            for index_string,tensor in itertools.izip(index_strings,tensors):
                slot = index_string.find(index)
                if slot >= 0:
                    return tensor,slot
            assert False, 'This should never happen.'
        lookup = tuple(tensor_and_slot_in_which_index_occurs(index) for index in index_string)
        return tuple(tensor.shape[slot] for tensor,slot in lookup)

    contraction_dims = dims_of_index_string(contraction_index_string)
    output_dims = dims_of_index_string(output_index_string)

    #
    # Starting here is the actual contraction computation
    #

    def component_indices_function (index_string):
        is_contraction_index = tuple(index in contraction_index_string for index in index_string)
        lookups = tuple((0 if is_contraction_index[i] else 1, contraction_index_string.index(index) if is_contraction_index[i] else output_index_string.index(index)) for i,index in enumerate(index_string))
        
        index_string_pair = (contraction_index_string, output_index_string)
        for i,lookup in enumerate(lookups):
            assert index_string[i] == index_string_pair[lookup[0]][lookup[1]]
            
        def component_indices_of (contracted_and_output_indices_tuple):
            assert len(lookups) == len(index_string)
            assert len(contracted_and_output_indices_tuple) == 2
            assert len(contracted_and_output_indices_tuple[0]) == len(contraction_index_string)
            assert len(contracted_and_output_indices_tuple[1]) == len(output_index_string)
            retval = tuple(contracted_and_output_indices_tuple[lookup[0]][lookup[1]] for lookup in lookups)
            return retval

        test_output = ''.join(component_indices_of((contraction_index_string, output_index_string)))
        assert test_output == index_string
        return component_indices_of
    
    component_indices_functions = tuple(component_indices_function(index_string) for index_string in index_strings)

    def product_of_components_of_tensors (contracted_and_output_indices_tuple):
        return reduce(operator.mul, tuple(tensor_component(tensor,component_indices_function(contracted_and_output_indices_tuple)) for tensor,component_indices_function in itertools.izip(tensors,component_indices_functions)), 1)

    def component (output_component_indices):
        return sum(product_of_components_of_tensors((contraction_component_indices, output_component_indices)) for contraction_component_indices in multiindex_iterator(contraction_dims))

    retval = np.ndarray(output_dims, dtype=dtype, buffer=np.array([component(output_component_indices) for output_component_indices in multiindex_iterator(output_dims)]))
    # If the result is a 0-tensor, then coerce it to the scalar type.
    if retval.shape == tuple():
        retval = retval[tuple()]
    return retval

def contract__run_unit_tests ():
    import symbolic
    import sympy
    import sys
    import traceback
    
    # Define a bunch of tensors to use in the tests
    x = sympy.symbols('x')
    T_ = symbolic.tensor('z', tuple())
    T_4 = symbolic.tensor('a', (4,))
    T_5 = symbolic.tensor('b', (5,))
    U_5 = symbolic.tensor('c', (5,))
    T_3_5 = symbolic.tensor('d', (3,5))
    T_4_3 = symbolic.tensor('e', (4,3))
    T_4_4 = symbolic.tensor('f', (4,4))
    T_5_2 = symbolic.tensor('g', (5,2))
    T_3_4_5 = symbolic.tensor('h', (3,4,5))
    T_3_3_4 = symbolic.tensor('i', (3,3,4))

    def is_zero_tensor (T):
        return all(t == 0 for t in T.flat) if hasattr(T,'shape') else (T == 0)

    def positive__unit_test_0a ():
        output_shape = (3,5,3)
        contraction_shape = (4,)
        expected_result = np.ndarray(output_shape, dtype=object, buffer=np.array([sum(T_3_4_5[i,j,k]*T_4_3[j,l] for (j,) in multiindex_iterator(contraction_shape)) for i,k,l in multiindex_iterator(output_shape)]))
        assert is_zero_tensor(contract('ijk,jl', T_3_4_5, T_4_3, dtype=object) - expected_result)
    def positive__unit_test_0b ():
        output_shape = (3,5,3)
        contraction_shape = (4,)
        expected_result = np.ndarray(output_shape, dtype=object, buffer=np.array([sum(T_3_4_5[i,j,k]*T_4_3[j,l] for (j,) in multiindex_iterator(contraction_shape)) for i,k,l in multiindex_iterator(output_shape)]))
        assert is_zero_tensor(contract('ijk,jl', T_3_4_5, T_4_3, output='ikl', dtype=object) - expected_result)
    def positive__unit_test_0c ():
        output_shape = (3,3,5)
        contraction_shape = (4,)
        expected_result = np.ndarray(output_shape, dtype=object, buffer=np.array([sum(T_3_4_5[i,j,k]*T_4_3[j,l] for (j,) in multiindex_iterator(contraction_shape)) for i,l,k in multiindex_iterator(output_shape)]))
        assert is_zero_tensor(contract('ijk,jl', T_3_4_5, T_4_3, output='ilk', dtype=object) - expected_result)
        
    def positive__unit_test_1a ():
        output_shape = (5,)
        contraction_shape = (3,4)
        expected_result = np.ndarray(output_shape, dtype=object, buffer=np.array([sum(T_3_4_5[i,j,k]*T_4_3[j,i] for i,j in multiindex_iterator(contraction_shape)) for (k,) in multiindex_iterator(output_shape)]))
        assert is_zero_tensor(contract('ijk,ji', T_3_4_5, T_4_3, dtype=object) - expected_result)
    def positive__unit_test_1b ():
        output_shape = (5,)
        contraction_shape = (3,4)
        expected_result = np.ndarray(output_shape, dtype=object, buffer=np.array([sum(T_3_4_5[i,j,k]*T_4_3[j,i] for i,j in multiindex_iterator(contraction_shape)) for (k,) in multiindex_iterator(output_shape)]))
        assert is_zero_tensor(contract('ijk,ji', T_3_4_5, T_4_3, output='k', dtype=object) - expected_result)
        
    def positive__unit_test_2a ():
        output_shape = tuple()
        contraction_shape = (5,)
        expected_result = np.ndarray(output_shape, dtype=object, buffer=np.array([sum(T_5[i]*T_5[i] for (i,) in multiindex_iterator(contraction_shape))]))
        assert is_zero_tensor(contract('i,i', T_5, T_5, dtype=object) - expected_result)
    def positive__unit_test_2b ():
        output_shape = tuple()
        contraction_shape = (5,)
        expected_result = np.ndarray(output_shape, dtype=object, buffer=np.array([sum(T_5[i]*T_5[i] for (i,) in multiindex_iterator(contraction_shape))]))
        assert is_zero_tensor(contract('i,i', T_5, T_5, output='', dtype=object) - expected_result)
    
    def positive__unit_test_3a ():
        output_shape = (5,5)
        contraction_shape = tuple()
        expected_result = np.ndarray(output_shape, dtype=object, buffer=np.array([T_5[i]*U_5[j] for i,j in multiindex_iterator(output_shape)]))
        assert is_zero_tensor(contract('i,j', T_5, U_5, dtype=object) - expected_result)
    def positive__unit_test_3b ():
        output_shape = (5,5)
        contraction_shape = tuple()
        expected_result = np.ndarray(output_shape, dtype=object, buffer=np.array([T_5[j]*U_5[i] for i,j in multiindex_iterator(output_shape)]))
        assert is_zero_tensor(contract('j,i', T_5, U_5, dtype=object) - expected_result)
    def positive__unit_test_3c ():
        output_shape = (5,5)
        contraction_shape = tuple()
        expected_result = np.ndarray(output_shape, dtype=object, buffer=np.array([T_5[i]*U_5[j] for i,j in multiindex_iterator(output_shape)]))
        assert is_zero_tensor(contract('i,j', T_5, U_5, output='ij', dtype=object) - expected_result)
    def positive__unit_test_3d ():
        output_shape = (5,5)
        contraction_shape = tuple()
        expected_result = np.ndarray(output_shape, dtype=object, buffer=np.array([T_5[i]*U_5[j] for j,i in multiindex_iterator(output_shape)]))
        assert is_zero_tensor(contract('i,j', T_5, U_5, output='ji', dtype=object) - expected_result)
    
    def positive__unit_test_4a ():
        output_shape = (4,2)
        contraction_shape = (3,5)
        expected_result = np.ndarray(output_shape, dtype=object, buffer=np.array([sum(T_4_3[i,j]*T_3_5[j,k]*T_5_2[k,l] for j,k in multiindex_iterator(contraction_shape)) for i,l in multiindex_iterator(output_shape)]))
        assert is_zero_tensor(contract('ij,jk,kl', T_4_3, T_3_5, T_5_2, dtype=object) - expected_result)
    def positive__unit_test_4b ():
        output_shape = (2,4)
        contraction_shape = (3,5)
        expected_result = np.ndarray(output_shape, dtype=object, buffer=np.array([sum(T_4_3[i,j]*T_3_5[j,k]*T_5_2[k,l] for j,k in multiindex_iterator(contraction_shape)) for l,i in multiindex_iterator(output_shape)]))
        assert is_zero_tensor(contract('lj,jk,ki', T_4_3, T_3_5, T_5_2, dtype=object) - expected_result)
    def positive__unit_test_4c ():
        output_shape = (4,2)
        contraction_shape = (3,5)
        expected_result = np.ndarray(output_shape, dtype=object, buffer=np.array([sum(T_4_3[i,j]*T_3_5[j,k]*T_5_2[k,l] for j,k in multiindex_iterator(contraction_shape)) for i,l in multiindex_iterator(output_shape)]))
        assert is_zero_tensor(contract('ij,jk,kl', T_4_3, T_3_5, T_5_2, output='il', dtype=object) - expected_result)
    
    def positive__unit_test_5a ():
        output_shape = tuple()
        contraction_shape = (4,)
        expected_result = np.ndarray(output_shape, dtype=object, buffer=np.array([sum(T_4_4[i,i] for (i,) in multiindex_iterator(contraction_shape))]))
        assert is_zero_tensor(contract('ii', T_4_4, dtype=object) - expected_result)
    def positive__unit_test_5b ():
        output_shape = tuple()
        contraction_shape = (4,)
        expected_result = np.ndarray(output_shape, dtype=object, buffer=np.array([sum(T_4_4[i,i] for (i,) in multiindex_iterator(contraction_shape))]))
        assert is_zero_tensor(contract('ii', T_4_4, output='', dtype=object) - expected_result)
    
    def positive__unit_test_6a ():
        output_shape = (4,)
        contraction_shape = (3,)
        expected_result = np.ndarray(output_shape, dtype=object, buffer=np.array([sum(T_3_3_4[i,i,j] for (i,) in multiindex_iterator(contraction_shape)) for (j,) in multiindex_iterator(output_shape)]))
        assert is_zero_tensor(contract('iij', T_3_3_4, dtype=object) - expected_result)
    def positive__unit_test_6b ():
        output_shape = (4,)
        contraction_shape = (3,)
        expected_result = np.ndarray(output_shape, dtype=object, buffer=np.array([sum(T_3_3_4[i,i,j] for (i,) in multiindex_iterator(contraction_shape)) for (j,) in multiindex_iterator(output_shape)]))
        assert is_zero_tensor(contract('iij', T_3_3_4, output='j', dtype=object) - expected_result)
    
    def positive__unit_test_7a ():
        expected_result = T_*T_
        assert is_zero_tensor(contract(',', T_, T_, dtype=object) - expected_result)
    def positive__unit_test_7b ():
        expected_result = T_*x
        assert is_zero_tensor(contract(',', T_, x, dtype=object) - expected_result)
    def positive__unit_test_7c ():
        expected_result = T_*x
        assert is_zero_tensor(contract(',', x, T_, dtype=object) - expected_result)
    def positive__unit_test_7d ():
        expected_result = x*x
        assert is_zero_tensor(contract(',', x, x, dtype=object) - expected_result)
    
    def positive__unit_test_8a ():
        assert is_zero_tensor(contract('', T_, dtype=object) - T_)
    def positive__unit_test_8b ():
        assert is_zero_tensor(contract('', x, dtype=object) - x)
    
    def positive__unit_test_9a ():
        # We will allow summation over indices that occur more than twice, even though
        # this indicates a type error in tensorial constructions.  But here, we're just
        # working with tensor-like grids of values, so no such assumption will be made.
        # Perhaps a warning could be printed, which could be turned off by the explicit
        # specification of a keyword argument.
        output_shape = tuple()
        contraction_shape = (5,)
        expected_result = np.ndarray(output_shape, dtype=object, buffer=np.array([sum(T_5[i]*T_5[i]*U_5[i] for (i,) in multiindex_iterator(contraction_shape))]))
        assert is_zero_tensor(contract('i,i,i', T_5, T_5, U_5, dtype=object) - expected_result)
    
    def negative__unit_test_0a ():
        contract('', T_5, T_4_4, dtype=object) # Wrong number of index strings.
    def negative__unit_test_0b ():
        contract('i,j,k', T_5, T_4_4, dtype=object) # Wrong number of index strings.
    def negative__unit_test_0c ():
        contract('i,j,k', T_4_4, dtype=object) # Wrong number of index strings.
    def negative__unit_test_0d ():
        contract('i,j', dtype=object) # Wrong number of index strings.

    def negative__unit_test_1a ():
        contract('', T_5, dtype=object) # Mismatch of number of indices and tensor order.
    def negative__unit_test_1b ():
        contract('ij', T_5, dtype=object) # Mismatch of number of indices and tensor order.
    def negative__unit_test_1c ():
        contract('ij', T_3_4_5, dtype=object) # Mismatch of number of indices and tensor order.

    def negative__unit_test_2a ():
        contract('i,i', T_5, T_4, dtype=object) # Non-matching contraction dimensions.
    def negative__unit_test_2b ():
        contract('i,i,i', T_5, T_4, T_4, dtype=object) # Non-matching contraction dimensions.
    def negative__unit_test_2c ():
        contract('ij,jk', T_4_3, T_4_4, dtype=object) # Non-matching contraction dimensions.
    def negative__unit_test_2d ():
        contract('ij,ij', T_4_3, T_4_4, dtype=object) # Non-matching contraction dimensions.
    def negative__unit_test_2e ():
        contract('ij,ij', T_5_2, T_4_4, dtype=object) # Non-matching contraction dimensions.

    def negative__unit_test_3a ():
        contract('ij,jk', T_4_3, T_3_5, output='ii', dtype=object)

    # Run all unit tests in alphabetical order.  The set of unit tests is defined
    # to be the set of callable local objects (see locals()), where an object obj is
    # callable iff hasattr(obj,'__call__') returns True.
    unit_test_count = 0
    pass_count = 0
    fail_count = 0
    for name in sorted(locals().keys()):
        obj = locals()[name]
        if hasattr(obj,'__call__'):
            # Positive and negative tests are run differently.
            if 'positive' in name:
                assert 'negative' not in name, 'Exactly one of the strings \'positive\' and \'negative\' should be present in a unit test name (in particular, the failing name is \'{0}\').'.format(name)
                unit_test_count += 1
                sys.stdout.write('Running {0} ... '.format(name))
                try:
                    obj()
                    sys.stdout.write('passed (no exception was raised).\n')
                    pass_count += 1
                except Exception as e:
                    sys.stdout.write('FAILED -- exception was {0}, stack trace was\n{1}\n'.format(repr(e), traceback.format_exc()))
                    fail_count += 1
            elif 'negative' in name:
                assert 'positive' not in name, 'Exactly one of the strings \'positive\' and \'negative\' should be present in a unit test name (in particular, the failing name is \'{0}\').'.format(name)
                unit_test_count += 1
                sys.stdout.write('Running {0} ... '.format(name))
                try:
                    obj() # In a negative test, we expect an exception to be raised.
                    sys.stdout.write('FAILED (expected exception to be raised in negative test, but none was raised).\n')
                    fail_count += 1
                except Exception as e:
                    sys.stdout.write('passed (caught expected exception {0}).\n'.format(repr(e)))
                    pass_count += 1
    if unit_test_count > 0:
        print 'Summary: {0} unit tests, {1} passed, {2} failed, failure rate was {3}%'.format(unit_test_count, pass_count, fail_count, float(fail_count)*100.0/unit_test_count)

if __name__ == '__main__':
    print 'Because this module is being run as \'__main__\', the unit tests will be run.'
    contract__run_unit_tests()
