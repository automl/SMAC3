"""
This file is overly documented and in general you don't need this much. It's good for
front facing API though.
"""
from typing import Any, Dict, List, Optional


class MyClass:
    """A short description of the class

    You can add more description here, that spans more lines, and takes up more space.
    But don't feel obliged, sometimes one line is enough...

    Parameters
    ----------
    a : int
        A description for parameter a

    b : Dict[str, Any]
         A description for parameter b

    c : bool = False
        A description for the parameter c with a default of False

    d : Optional[List[float]] = None
         A description for the optional parameter c
    """

    def __init__(
        self,
        a: int,
        b: Dict[str, Any],
        c: bool = False,
        d: Optional[List[float]] = None,
    ):
        if a < 1:
            raise ValueError("MyClass(a) must be a positive integer")

        self.a = a
        self.b = b
        self.c = c
        self.d = d

    @property
    def spaghettihoops(self) -> Dict[str, str]:
        """Spaghetti hoops of this object

        Has as many entries as the parameter ``a`` given at construction. If ``c`` is
        True, the values will have "foo", otherwise "bar".

        .. code:: python

            myclass = MyClass(a=2, b={}, c=True)
            print(myclass.spaghetti_hoops)
            {
                "hoop_0": "foo",
                "hoop_1": "foo",
            }
            myclass = MyClass(a=2, b={}, c=False)
            print(myclass.spaghetti_hoops)
            {
                "hoop_0": "bar",
                "hoop_1": "bar",
            }

        Returns
        -------
        Dict[str, str]
            A dictionary from "hoop_i" to its value
        """
        return {f"hoop_{i}": "foo" if self.c else "bar" for i in range(self.a)}

    def oreos(self, x: int) -> int:
        """A function that does something

        Here is a longer description which might give some more details or even
        an example

        .. code:: python

            result = myclass.some_function(42)

        Note
        ----
        Something you should defnitely pay attention to

        Parameters
        ----------
        x : int
            Some description about x

        Returns
        -------
        int
            The sum of x and a

        Raises
        ------
        ValueError
            This function really doesn't allow for the number 1337
        """
        if x == 1337:
            raise ValueError("Leet HaX04 detected")

        return x + self.a
