"""
Test that the wrapped tokenizer loaded provides the same outputs
as the original tokenizer (i.e. just using the token map in cfg)

We may want a separate set of tests for different tokenization schemes
"""
# TODO - Pending tokenizer


def test_tokenization_encoding():
    # Check that wrapped tokenizer __call__ returns the same as original tokenizer
    pass

def test_tokenization_decoding():
    # Check that wrapped tokenizer decode returns the same as iterating through token map
    pass

def test_padding():
    # Check that wrapped tokenizer pad returns the same as original padding function 
    pass

def test_inside_hooked_transformer():
    # Check that the wrapped tokenizer facilitates all HookedTransformer tokenizer-dependent functionality
    pass
