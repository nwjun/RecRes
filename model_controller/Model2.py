from abc import abstractmethod, ABC


class Model2(ABC):
    @abstractmethod
    def get_cuisine(self, ingredients) -> str:
        pass

    @abstractmethod
    def get_recipes(self, ingredients, cuisines) -> str:
        pass