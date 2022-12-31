import random
import sys

import numpy
import numpy as np


# ========================================= Graph data =========================================

class Graph:
    # stwórz instancję grafu
    def __init__(self, matrix: np.ndarray, alpha: float, beta: float, rho: float, q: float):
        self.matrix = matrix
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.q = q
        self.n = matrix.shape[0]
        self.optimal_value = sys.maxsize
        self.pheromone_intensity = np.full((self.n, self.n), self.n / nearest_neighbour(self.matrix))

    # odśwież intensywność feromonów na wszystkich wierzchołkach
    def update_pheromones(self, new_pheromones: list[tuple]):
        # odparuj feromony
        self._evaporate_pheromones()

        # zwiększ ilość feromonów na krawędzach przebytych przez mrówki według wzoru algorytmu QAS (Wzór 4)
        for pair in new_pheromones:
            self.pheromone_intensity[pair[0]][pair[1]] += self.q / self.matrix[pair[0]][pair[1]]

    def _evaporate_pheromones(self):
        self.pheromone_intensity = np.multiply(self.pheromone_intensity, self.rho)

    def get_edge_info(self, current: int, new: int) -> tuple[int, float]:
        return self.matrix[current][new], self.pheromone_intensity[current][new]

    def pick_next_node(self, current: int, not_visited: list):
        p = list()
        for node in not_visited:
            edge_weight = 0.1
            if self.matrix[current][node] != 0:
                edge_weight = self.matrix[current][node]
            p.append((max(self.pheromone_intensity[current][node], 1e-5) ** self.alpha) / (edge_weight ** self.beta))

        return random.choices(not_visited, weights=p)

    def revalidate_optimal(self, path: list) -> bool:
        path_value = self._evaluate_path(path)

        if self.optimal_value > path_value:
            self.optimal_value = path_value
        elif self.optimal_value == path_value:
            return True
        return False

    def _evaluate_path(self, path: list) -> int:
        path_value = 0
        for i in range(len(path) - 1):
            path_value += self.matrix[path[i]][path[i+1]]

        return path_value


# ===================================== Ant population data ====================================

class AntColony:
    # stwórz generację mrówek
    def __init__(self, n: int):
        self.n = n
        self.paths = np.empty(n, dtype=list)
        self.not_visited = np.empty(n, dtype=list)

        for i in range(n):
            if self.paths[i] is not None:
                self.paths[i].append(i)
            else:
                self.paths[i] = list([i])
            self.not_visited[i] = list([j for j in range(n) if j != i])

    # przemieść mrówki
    def move_all(self, graph: Graph) -> list[tuple]:
        moves = list()
        # rób dopóki licznik 'ant+no' jest mniejszy od liczby mrówek
        for ant_no in range(self.n):
            # wybierz losowo kolejny wierzchołek dla mrówki z prawdopodobieństwem opisanym wzorem (Wzór 2)
            next_node = graph.pick_next_node(self.paths[ant_no][-1], self.not_visited[ant_no])[0]

            # dodaj ruch z wierzchołka aktualnego do nowego wylosowanego do listy ruchów
            moves.append(tuple([self.paths[ant_no][-1], next_node]))

            self.paths[ant_no].append(next_node)  # dodaj wylosowany wierzchołek do ścieżki mrówki
            self.not_visited[ant_no].remove(next_node)  # usuń wylosowany wierzchołek z listy nieodwiedzonych

        # zwróć listę ruchów mrówek
        return moves

    # sprawdź ścieżki pokonane przez mrówki
    def check_population_paths(self, graph):
        unipath_counter = 0

        for path in self.paths:  # rób dla wszystkich ścieżek mrówek
            path.append(path[0])
            if graph.revalidate_optimal(path):
                unipath_counter += 1

        return unipath_counter == graph.n


def aco(array: np.ndarray, alpha: float, beta: float, rho: float, q: float, ants_to_stop: int) -> int:
    n = array.shape[0]
    cycles = numpy.ceil(ants_to_stop / n)  # oblicz liczbę generacji mrówek

    # stwórz instancję grafu zawierającą tablicę sąsiedztwa, tablicę intensywności feromonów
    graph = Graph(array, alpha, beta, rho, q)

    for cycle_no in range(int(cycles)):  # rób dopóki licznik 'cycle_no' mniejszy od liczby generacji mrówek
        ant_generation = AntColony(n)  # stwórz generację mrówek
        for i in range(n - 1):  # rób dopóki licznik 'i' jest mniejszy od wielkości instancji pomniejszonej o 1
            new_pheromones = ant_generation.move_all(graph)  # przemieść mrówki do kolejnych nieodwiedzonych wierzchołków
            graph.update_pheromones(new_pheromones)  # odśwież intensywność feromonów na wszystkich wierzchołkach

        # sprawdź ścieżki pokonane przez mrówki
        if ant_generation.check_population_paths(graph):  # jeżeli mrówki miały równe wartości ścieżek
            break

    return graph.optimal_value  # zwróć optymalną wartość ścieżki


def nearest_neighbour(array: np.ndarray) -> int:
    # dodaj wierzchołek początkowy (0) do listy odwiedzonych, oraz jako aktualny wierzchołek
    n, visited, path_value, current_node = array.shape[0], list([0]), 0, 0

    # rób dopóki długość listy odwiedzonych jest mniejsza od wielkości instancji
    while len(visited) < n:
        # znajdź najbliższego sąsiada aktualnego wierzchołka, należącego do wierzchołków nieodwiedzonych
        lowest, nearest_node = sys.maxsize, None

        for i in [i for i in range(n) if i not in visited]:
            if array[current_node][i] < lowest:
                lowest, nearest_node = array[current_node][i], i

        path_value += lowest  # dodaj krawędź między aktualnym a wyznaczonym wierzchołkiem do wartości ścieżki
        visited.append(nearest_node)  # dodaj wyznaczony wierzchołek do listy odwiedzonych

        current_node = nearest_node  # ustaw wyznaczony wierzchołek jako nowy aktualny

    visited.append(0)  # dodaj powrót z ostatniego wierzchołka do początkowego do ścieżki
    # dodaj wartość krawędzi z ostatniego wierzchołka do początkowego do wartości ścieżki
    path_value += array[current_node][0]

    return path_value  # zwróć wartość ścieżki
