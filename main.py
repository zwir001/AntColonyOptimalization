# PEA - projekt, WT 15:15
# Zadanie 3 - TSP - Algorytm Mrówkowy (ACO)
# Jakub Wirwis 259128
from time import perf_counter

from babel.numbers import format_scientific
from memory_profiler import memory_usage
from ant_colony_optimization import aco

import helpers as h

if __name__ == '__main__':
    config = h.read_config()  # odczytaj plik ini
    config.pop(0)

    results_csv = list()

    config_data_line = config.pop(0)
    results_csv.append(config_data_line)
    alpha, beta, rho = float(config_data_line[0]), float(config_data_line[1]), float(config_data_line[2])

    for line in config:
        if len(line) > 1:
            results_csv.append(line)

            print(line[0])

            if int(line[1]) != 0:
                distances_matrix = h.read_instance_format_tsp(line[0])
                instance_size = distances_matrix.shape[0]

                for repeat in range(int(line[1])):
                    print(repeat)
                    start = perf_counter()  # rozpocznij pomiar czasu
                    result = aco(distances_matrix, alpha, beta, rho, 10, 300)
                    end = perf_counter()  # zakończ pomiar czasu
                    mem_usage = memory_usage((aco, (distances_matrix, alpha, beta, rho, 100, 5)))
                    results_csv.append(
                        [
                            format_scientific(end - start, locale="pl"),
                            format_scientific(max(mem_usage), locale="pl"),
                            format_scientific(result, locale="pl")
                        ]
                    )
        else:
            h.write_to_csv(line[0], results_csv)
