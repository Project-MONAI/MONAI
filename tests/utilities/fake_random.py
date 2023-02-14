
class FakeRandomState:

    def __init__(self, value_pairs):
        self.index = 0
        self.value_pairs = value_pairs

    @staticmethod
    def __check_compatible(index, function, argument):
        if argument[0] != function:
            raise ValueError(f"FakeRandomState value at index {index} is for "
                             f"'{argument[0]}' but '{function}' was called")

    def rand(self, *_):
        self.__check_compatible(self.index, 'rand', self.value_pairs[self.index])
        value = self.value_pairs[self.index][1]
        if isinstance(value, Exception):
            raise value
        self.index += 1
        return value

    def uniform(self, *_):
        self.__check_compatible(self.index, 'uniform', self.value_pairs[self.index])
        value = self.value_pairs[self.index][1]
        self.index += 1
        if isinstance(value, Exception):
            raise value
        return value
