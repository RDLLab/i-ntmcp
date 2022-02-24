"""Base state class """


class State:
    """An abstract state class """

    def render_asci(self):
        """Can be overwritten in child classes to display more interesting
        state representations (e.g. a grid).
        """
        return self.__str__()

    def __hash__(self):
        raise NotImplementedError

    def __eq__(self, other):
        raise NotImplementedError

    def __repr__(self):
        return self.__str__()


class DiscreteState(State):
    """A basic discrete state class """

    def __init__(self, state_num: int):
        super().__init__()
        self.state_num = state_num

    def copy(self):
        """Return deepcopy of state """
        return self.__class__(self.state_num)

    def __str__(self):
        return f"{self.state_num}"

    def __hash__(self):
        return self.state_num

    def __eq__(self, other):
        return self.state_num == other.state_num
