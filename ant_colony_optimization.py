import random
import sys
import numpy as np


# ========================================= Graph data =========================================

class Graph:
    def __init__(self, matrix: np.ndarray, alpha: float, beta: float, rho: float, q: float):
        self.matrix = matrix
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.q = q
        self.n = matrix.shape[0]
        self.optimal_value = nearest_neighbour(matrix)
        self.pheromone_intensity = np.full((self.n, self.n), self.n / self.optimal_value)

    def update_pheromones(self, new_pheromones: list[tuple]):
        self._evaporate_pheromones()
        for pair in new_pheromones:
            self.pheromone_intensity[pair[0]][pair[1]] += self.q

    def _evaporate_pheromones(self):
        self.pheromone_intensity = np.multiply(self.pheromone_intensity, 0.5)

    def get_edge_info(self, current: int, new: int) -> tuple[int, float]:
        return self.matrix[current][new], self.pheromone_intensity[current][new]

    def pick_next_node(self, current: int, not_visited: list):
        p = list()
        for node in not_visited:
            p.append((self.pheromone_intensity[current][node] ** self.alpha) / (self.matrix[current][node] ** self.beta))

        return random.choices(not_visited, weights=p)

    def revalidate_optimal(self, path: list) -> bool:
        path_value = self._evaluate_path(path)

        if self.optimal_value > path_value:
            self.optimal_value = path_value
        elif self.optimal_value == path_value:
            return False
        return True

    def _evaluate_path(self, path: list) -> int:
        path_value = 0
        for i in range(len(path) - 1):
            path_value += self.matrix[path[i]][path[i+1]]

        return path_value


# ===================================== Ant population data ====================================

class AntColony:
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

    def move_all(self, graph: Graph) -> list[tuple]:
        moves = list()
        for ant_no in range(self.n):
            next_node = graph.pick_next_node(self.paths[ant_no][-1], self.not_visited[ant_no])[0]

            self.paths[ant_no].append(next_node)
            self.not_visited[ant_no].remove(next_node)
            moves.append(tuple([self.paths[ant_no][-1], next_node]))

        return moves

    def check_population_paths(self, graph):
        unipath_counter = 0
        for path in self.paths:
            path.append(path[0])
            if not graph.revalidate_optimal(path):
                unipath_counter += 1

        if unipath_counter > graph.n * 0.4:
            return False
        print(unipath_counter)
        return True


def aco(array: np.ndarray, alpha: float, beta: float, rho: float, q: float, cycles_to_stop: int) -> int:
    n = array.shape[0]
    graph = Graph(array, alpha, beta, rho, q)
    for cycle_no in range(cycles_to_stop):
        ant_generation = AntColony(array.shape[0])
        for i in range(n - 1):
            new_pheromones = ant_generation.move_all(graph)
            graph.update_pheromones(new_pheromones)

        if not ant_generation.check_population_paths(graph):
            break

    return graph.optimal_value


def nearest_neighbour(array: np.ndarray) -> int:
    n, visited, path_value, current_node = array.shape[0], list([0]), 0, 0

    while len(visited) < n:
        lowest, nearest_node = sys.maxsize, None

        for i in [i for i in range(n) if i not in visited]:
            if array[current_node][i] < lowest:
                lowest, nearest_node = array[current_node][i], i

        path_value += lowest
        visited.append(nearest_node)

        current_node = nearest_node

    visited.append(0)
    path_value += array[current_node][0]

    return path_value
