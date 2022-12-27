import csv

import networkx
import numpy as np
import tsplib95


def read_config():
    parameters = []
    file = open("config.ini", "r")
    file_lines = file.read().splitlines()

    for line in file_lines:
        instance_parameters = line.split(',')
        parameters.append(instance_parameters)

    return parameters


def read_instance_format_tsp(file_name) -> np.matrix:
    problem = tsplib95.load("instances/" + file_name)

    graph = problem.get_graph()

    distance_matrix = networkx.to_numpy_array(graph)

    return distance_matrix


def write_to_csv(file_name, results):
    with open(file_name, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(results)
