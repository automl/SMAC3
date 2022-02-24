import sys

__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"


if __name__ == '__main__':

    if sys.argv[1] == "1":
        print("Result of this algorithm run: UNSAT,1,1,1,12354")

    if sys.argv[1] == "2":
        print("Result of this algorithm run: UNSAT,2,3,4,12354,additional info")
