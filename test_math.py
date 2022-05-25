from parameterized import parameterized

@parameterized([
    (1, 2, 3),
    (4, 5, 9),
])
def test_add(a, b, expected):
    """ Test addition. """
    assert_equal(a + b, expected)

def my_doc_func(func, num, param):
    return "%s: %s with %s" %(num, func.__name__, param)

@parameterized([
    (5, 4, 1),
    (9, 6, 3),
], doc_func=my_doc_func)
def test_subtraction(a, b, expected):
    assert_equal(a - b, expected)
