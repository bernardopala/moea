import os
from decimal import Decimal
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
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

def to_scientific_notation(value):
    return f"{Decimal(value):.4e}"

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

def generate_evaluation_summary_html_table():
    df = generate_evaluation_summary_df('results/comparative_analysis/')
    df.loc[df['Algorytm'] == 'NSGAII', 'Algorytm'] = 'NSGA-II'
    df.loc[df['Algorytm'] == 'MOEAD', 'Algorytm'] = 'MOEA/D'
    df.loc[df['Algorytm'] == 'Epsilon-IBEA', 'Algorytm'] = 'IBEA'

    problems_order = ['Kursawe', 'Binh2', 'DTLZ2', 'DTLZ7']
    algorithms_order = ['NSGA-II', 'SPEA2', 'MOEA/D', 'IBEA']
    df['Nazwa problemu'] = pd.Categorical(df['Nazwa problemu'], categories=problems_order, ordered=True)
    df['Algorytm'] = pd.Categorical(df['Algorytm'], categories=algorithms_order, ordered=True)
    df_sorted = df.sort_values(['Nazwa problemu', 'Algorytm']).reset_index(drop=True)

    best_hv = df_sorted.groupby('Nazwa problemu', observed=False)['HV (średnia)'].idxmax()
    best_igd = df_sorted.groupby('Nazwa problemu', observed=False)['IGD (średnia)'].idxmin()

    df_sorted['HV (średnia)'] = df_sorted['HV (średnia)'].apply(to_scientific_notation)
    df_sorted['HV (odchylenie std.)'] = df_sorted['HV (odchylenie std.)'].apply(to_scientific_notation)
    df_sorted['IGD (średnia)'] = df_sorted['IGD (średnia)'].apply(to_scientific_notation)
    df_sorted['IGD (odchylenie std.)'] = df_sorted['IGD (odchylenie std.)'].apply(to_scientific_notation)

    html = df_sorted.to_html(escape=False, index=False).replace('class="dataframe"', '')

    # pogrubienie najlepszych wyników
    soup = BeautifulSoup(html, 'html.parser')
    rows = soup.find('tbody').find_all('tr')

    for row_idx in best_hv:
        tds = rows[row_idx].find_all('td')

        b_tag = soup.new_tag("b")
        b_tag.string = tds[2].text
        tds[2].clear()
        tds[2].append(b_tag)

        b_tag = soup.new_tag("b")
        b_tag.string = tds[3].text
        tds[3].clear()
        tds[3].append(b_tag)

    for row_idx in best_igd:
        tds = rows[row_idx].find_all('td')

        b_tag = soup.new_tag("b")
        b_tag.string = tds[4].text
        tds[4].clear()
        tds[4].append(b_tag)

        b_tag = soup.new_tag("b")
        b_tag.string = tds[5].text
        tds[5].clear()
        tds[5].append(b_tag)

    html_modified = str(soup)

    return html_modified