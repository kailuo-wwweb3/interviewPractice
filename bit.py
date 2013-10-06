for n in range(17):
    if (n & 1):                   # if n is odd, n
        i = -((n + 1) >> 1)       # represents a negative number
    else:
        i = (n >> 1)
    print i