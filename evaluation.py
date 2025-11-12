import os
import numpy as np
import pandas as pd
from jmetal.core.quality_indicator import HyperVolume, NormalizedHyperVolume, InvertedGenerationalDistance
from pandas.core.interchange.dataframe_protocol import DataFrame

def get_algorithms_and_problems(folder):
    algorithms = []
    problems = []
    for filename in os.listdir(folder):
        algorithms.append(filename.split('.')[1])
        problems.append(filename.split('.')[2])
    return list(set(algorithms)), list(set(problems))

def find_files_with_phrase(folder, phrase):
    result = []
    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)
        if os.path.isfile(path) and phrase in filename:
            result.append(path)
    return result

def summarize_evaluation(problem_name, algorithm_name, indicator):
    folder = "results/comparative_analysis/"
    phrase = f"FUN.{algorithm_name}.{problem_name}"
    result_files = find_files_with_phrase(folder, phrase)

    ref_front = np.loadtxt(f"resources/reference_fronts/{problem_name}.pf")
    ref_front_dim = ref_front.shape[1]

    indicator_results = np.array([])
    for rf in result_files:
        fun = np.loadtxt(rf).reshape(-1, ref_front_dim)
        indicator_results = np.append(indicator_results, indicator.compute(fun))

    return np.mean(indicator_results), np.std(indicator_results)

def calculate_hv(problem_name, algorithm_name):
    ref_front = np.loadtxt(f"resources/reference_fronts/{problem_name}.pf")
    hv_ref_point = np.max(ref_front, axis=0) + 0.1
    hv = HyperVolume(hv_ref_point)

    return summarize_evaluation(problem_name, algorithm_name, hv)

def calculate_igd(problem_name, algorithm_name):
    ref_front = np.loadtxt(f"resources/reference_fronts/{problem_name}.pf")
    igd = InvertedGenerationalDistance(ref_front)

    return summarize_evaluation(problem_name, algorithm_name, igd)

def generate_evaluation_summary_df(folder):
    algorithms, problems = get_algorithms_and_problems(folder)

    results = np.array([])

    for problem in sorted(problems):
        for algorithm in sorted(algorithms):
            results = np.append(results, problem)
            results = np.append(results, algorithm)

            hv = calculate_hv(problem, algorithm)
            results = np.append(results, hv[0])
            results = np.append(results, hv[1])

            igd = calculate_igd(problem, algorithm)
            results = np.append(results, igd[0])
            results = np.append(results, igd[1])

    df = pd.DataFrame(results.reshape(-1, 6), columns=[ 'Nazwa problemu', 'Algorytm', 'HV (średnia)', 'HV (odchylenie std.)', 'IGD (średnia)', 'IGD (odchylenie std.)'])

    return df