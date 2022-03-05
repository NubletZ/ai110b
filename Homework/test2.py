def lin_equ(l1, l2):
    """Line encoded as l=(x,y)."""
    m = float((l2[1] - l1[1])) / float(l2[0] - l1[0])
    c = (l2[1] - (m * l2[0]))
    return m, c

# Example Usage:
a, b = lin_equ((2, 3.9,), (4, 6.2))
print("a={:f} b={:f}".format(a, b))