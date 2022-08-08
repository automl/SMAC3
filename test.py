from collections import namedtuple

Point = namedtuple("Point", "x y")
pt1 = Point(1.0, 5.0)
pt2 = Point(2.5, 1.5)


print(2.0 in pt1)
print(1.0 in pt1)
