from abc import abstractmethod, ABC


class Model(ABC):
    @abstractmethod
    def get_output(self, ingredients):
        pass

    @abstractmethod
    def format_output(self, output) -> str:
        pass
