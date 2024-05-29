

import math
from abc import ABC, abstractmethod


class Foundation(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def calculate_design_values(self):
        pass

    @abstractmethod
    def print_results(self):
        pass
