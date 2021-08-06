import sys

__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"


if __name__ == '__main__':

    if sys.argv[1] == "1":
        print('''Result of this algorithm run: {"status": "TIMEOUT", "cost": 1.0, "runtime": 2.0}''')

    if sys.argv[1] == "2":
        print('''Result of this algorithm run: {"status": \
        "SUCCESS", "cost": 2.0, "runtime": 3.0, "additional_info": "hello world!"}''')
