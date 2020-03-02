import sys

if __name__ == '__main__':
    # Unused in this example:
    # instance, instance_specific, cutoff, runlength = sys.argv[1:5]
    seed = sys.argv[5]
    # sys.argv[6] and sys.argv[8] are the names of the target algorithm
    # parameters (here: "-x1", "-x2")
    x1 = float(sys.argv[7])
    x2 = float(sys.argv[9])
    r1 = x1 % 0.9 - (x1 * 0.5) % 1.6
    r2 = - (x2 * 0.7) % 2.3 + (x2 * 1.5) % 3.5
    result = r1 + r2
    print('Result for SMAC: SUCCESS, -1, -1, %f, %s' % (result, seed))
