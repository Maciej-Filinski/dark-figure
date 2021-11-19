import pandas as pd
import numpy as np
import os
import datetime
import epiweeks
from module.dark_figure import DarkFigure


if __name__ == '__main__':
    cases_csv = 'https://raw.githubusercontent.com/KITmetricslab/covid19-forecast-hub-de/master/data-truth/RKI/by_age/truth_RKI-Incident%20Cases%20by%20Age_Germany.csv'
    cases = pd.read_csv(cases_csv)

    deaths_csv = 'https://raw.githubusercontent.com/KITmetricslab/covid19-forecast-hub-de/master/data-truth/RKI/by_age/truth_RKI-Incident%20Deaths%20by%20Age_Germany.csv'
    deaths = pd.read_csv(deaths_csv)
    deaths.loc[deaths['location_name'] == 'Free State of Thuringia', 'location_name'] = 'Free State of Thüringia'

    population_excel = './data/population_2020.xlsx'
    population = pd.read_excel(population_excel)
    population.columns = ['Age'] + list(population[3:4].to_numpy()[0][1:])
    population = population[5:-6].reset_index(drop=True)

    total_deaths_excel = 'https://www.destatis.de/DE/Themen/Gesellschaft-Umwelt/Bevoelkerung/Sterbefaelle-Lebenserwartung/Tabellen/sonderauswertung-sterbefaelle.xlsx;jsessionid=1A8D9AD59C7B0337F82EF6DBEB7D2749.live721?__blob=publicationFile'
    # for whole country use this:
    sheet_name = 'D_2016_2021_KW_AG_Ins'
    groupby = ['Alter von ... bis']
    # but for bundesland use this:
    sheet_name = 'BL_2016_2021_KW_AG_Ins'
    groupby = ['unter … Jahren']
    total_deaths = pd.read_excel(total_deaths_excel, sheet_name=sheet_name)
    total_deaths.columns = ['Nr', 'Year', 'Region', 'Age group'] + list(np.arange(1, 54))
    total_deaths = total_deaths[8:][total_deaths.columns[1:]].fillna(0).replace('X ', 0).reset_index(drop=True)

    population_excel = './data/population_2015-2019.xlsx'
    population_old = pd.read_excel(population_excel)
    # population_old = population_old[5:-6].reset_index(drop=True)
    population_old.columns = population.columns
    prior_populations = dict()
    step = len(population) + 1
    for start in range(4, 4 + step * 5, step):
        population_year = int((population_old[start:start + 1].to_numpy()[0][0])[-4:])
        df = population_old[start + 1:start + 1 + len(population)].reset_index(drop=True)
        df['Total'] = df[df.columns[1:]].apply(lambda x: sum(x), axis=1)
        df['Age'] = population['Age']
        df.index = df['Age']
        prior_populations[population_year] = df

    prior_populations = dict()
    step = len(population) + 1
    for start in range(4, 4 + step * 5, step):
        population_year = int((population_old[start:start + 1].to_numpy()[0][0])[-4:])
        df = population_old[start + 1:start + 1 + len(population)].reset_index(drop=True)
        df['Total'] = df[df.columns[1:]].apply(lambda x: sum(x), axis=1)
        df['Age'] = population['Age']
        df.index = df['Age']
        prior_populations[population_year] = df
    prior_populations

    population['Total'] = population[population.columns[1:]].apply(lambda x: sum(x), axis=1)
    population.index = population['Age']
    population

    locations1 = list(deaths['location_name'].unique())
    locations2 = ['Total', 'Baden-Württemberg', 'Bayern', 'Bremen', 'Hamburg', 'Hessen', 'Niedersachsen',
                  'Nordrhein-Westfalen', 'Rheinland-Pfalz', 'Saarland', 'Schleswig-Holstein',
                  'Brandenburg', 'Mecklenburg-Vorpommern',
                  'Sachsen', 'Sachsen-Anhalt', 'Thüringen', 'Berlin']
    all_locations = list(zip(locations1, locations2))
    print(all_locations)
    simulation = DarkFigure(deaths, cases)
    simulation(['Germany'])
