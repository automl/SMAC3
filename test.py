class A:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.a = "blub"

        # print(self.b)

    def get_a(self):
        return self.__class__.__name__


class B:
    def __init__(self, b):
        # print("B")
        self.b = b

    def get_a(self):
        return self.__class__.__name__


class C(A, B):
    def __init__(self):
        # super(A, self).__init__()
        # super(C, self).__init__()
        # super(B).__init__()
        super().__init__(b="blub")
        # print("C")


blub = C()

print(blub.get_a())
