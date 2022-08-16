class B:
    def __init__(self, text) -> None:
        self.haha2 = text

    pass


class C:
    def __init__(self) -> None:
        self.haha = "yooo"

    def yes(self):
        return self.blub


class A(B, C):
    def __init__(self) -> None:
        # super(C).__init__()
        B.__init__(self, "hey")
        C.__init__(self)
        self.blub = "YES"


classA = A()
print(classA.yes())
print(classA.haha)
