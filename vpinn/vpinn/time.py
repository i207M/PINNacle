from abc import ABC, abstractmethod

class time(ABC):
    @abstractmethod
    def generate_initial_points(self, num):
        pass