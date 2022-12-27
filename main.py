# PEA - projekt, WT 15:15
# Zadanie 3 - TSP - Algorytm Mrówkowy (ACO)
# Jakub Wirwis 259128
from time import perf_counter

from babel.numbers import format_scientific
from memory_profiler import memory_usage

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

            distances_matrix = h.read_instance_format_tsp(line[0])

            instance_size = distances_matrix.shape[0]

            for repeat in range(int(line[1])):
                start = perf_counter()  # rozpocznij pomiar czasu
                result = 0
                end = perf_counter()  # zakończ pomiar czasu

                mem_usage = memory_usage()
                results_csv.append(
                    [
                        format_scientific(end - start, locale="pl"),
                        format_scientific(max(mem_usage), locale="pl"),
                        format_scientific(result, locale="pl")
                    ]
                )
        else:
            h.write_to_csv(line[0], results_csv)
