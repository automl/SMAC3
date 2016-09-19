import sys
import numpy as np

def main():
    seed = sys.argv[5]
    x = float(sys.argv[7])
    y = float(sys.argv[9])
    tmp = branin((x, y))
    print('Result for SMAC: SUCCESS, -1, -1, %f, %s' % (tmp, seed))

import numpy as np

def branin(x):
    x1 = x[0]
    x2 = x[1]
    a = 1.
    b = 5.1 / (4.*np.pi**2)
    c = 5. / np.pi
    r = 6.
    s = 10.
    t = 1. / (8.*np.pi)
    ret = a*(x2-b*x1**2+c*x1-r)**2+s*(1-t)*np.cos(x1)+s
    return ret

if __name__ == '__main__':
    main()