# PEA - projekt, WT 15:15
# Zadanie 3 - TSP - Algorytm Mrówkowy (ACO)
# Jakub Wirwis 259128
import multiprocessing
from time import perf_counter

from babel.numbers import format_scientific
from memory_profiler import memory_usage
from ant_colony_optimization import aco

import helpers as h


def main():
    config = h.read_config()  # odczytaj plik ini
    config.pop(0)  # usuń nagłówek z opisem parametrów w pierwszej linii

    results_csv = list()

    # wczytaj parametry alpha, beta, rho z pliku konfiguracyjnego
    config_data_line = config.pop(0)
    results_csv.append(config_data_line)
    alpha, beta, rho = float(config_data_line[0]), float(config_data_line[1]), float(config_data_line[2])

    # rób dopóki plik zawiera kolejne linie
    for line in config:
        if len(line) > 1:  # jeżeli ilość wartości oddzielonych przecinkami w linii, większa od 1 to
            results_csv.append(line)  # zapisz linię do pliku wynikowego

            print(line[0])  # wypisz nazwę instancji

            if int(line[1]) != 0:  # jeżeli liczba wykonań jest różna od 0
                # wczytaj dane o instancji
                distances_matrix = h.read_instance_format_tsp(line[0])
                instance_size = distances_matrix.shape[0]

                # rób dopóki licznik 'repeat' jest mniejszy od liczby wykonań wykonań
                for repeat in range(int(line[1])):
                    print(repeat + 1)  # wypisz numer powtórzenia
                    start = perf_counter()  # rozpocznij pomiar czasu
                    result = aco(distances_matrix, alpha, beta, rho, 100, 1000)  # wywołaj algorytm
                    end = perf_counter()  # zakończ pomiar czasu

                    # dokonaj pomiaru pamięci
                    mem_usage = memory_usage((aco, (distances_matrix, alpha, beta, rho, 100, 1000)))

                    # zapisz zmierzony czas, maksymalne zużycie pamięci, oraz wynik algorytmu do pliku wynikowego
                    results_csv.append(
                        [
                            format_scientific(end - start, locale="pl"),
                            format_scientific(max(mem_usage), locale="pl"),
                            format_scientific(result, locale="pl")
                        ]
                    )
        else:
            h.write_to_csv(line[0], results_csv)  # zapisz plik wynikowy .csv


if __name__ == '__main__':
    multiprocessing.freeze_support()

    main()
