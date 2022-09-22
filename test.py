class A:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.a = "blub"

        # print(self.b)

    @property
    def meta(self):
        return self.a


blub = A()

print(blub.meta)
blub.a = "ajskedfö"
print(blub.meta)
