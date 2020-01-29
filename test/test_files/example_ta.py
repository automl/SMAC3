import sys


def sample(x):
    x1 = x[0]
    x2 = x[1]
    ret = x1 + x2
    return ret


if __name__ == '__main__':
    seed = sys.argv[5]
    x = float(sys.argv[7])
    y = float(sys.argv[9])
    tmp = sample((x, y))
    print('Result for SMAC: SUCCESS, -1, -1, %f, %s' % (tmp, seed))
