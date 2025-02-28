import math


class Species:
    def __init__(self):
        self.lat_name = ""
        self.eng_name = ""

        self.mean_territory_size = 20000 # in meter²
        self.__max_root_2_leaf_distance = 1.5 * math.sqrt((self.mean_territory_size) / math.pi)

    def max_root_2_leaf_distance(self):
        self.__max_root_2_leaf_distance = 1.5 * math.sqrt((self.mean_territory_size) / math.pi)
        return self.__max_root_2_leaf_distance