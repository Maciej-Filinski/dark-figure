import pandas as pd
import numpy as np
import os
import datetime
import epiweeks
from matplotlib import pyplot as plt


class DarkFigure:
    def __init__(self, deaths: pd.DataFrame, cases: pd.DataFrame, total_deaths: pd.DataFrame, population,
                 prior_populations):
        self.deaths = deaths
        self.cases = cases
        self.total_deaths = total_deaths
        self.population = population
        self.prior_populations = prior_populations
        self.all_regions = None

    def __call__(self, all_regions: list, ifr, ratios=None):
        self.all_regions = all_regions
        for region in all_regions:
            if region != 'Free State of Saxony':
                continue
            self.result_dir = f'./result/{str(datetime.datetime.now().date())}/{all_regions[0]}/{region}/'
            if not os.path.exists(self.result_dir):
                os.makedirs(self.result_dir)
            daily_regional_deaths, weekly_regional_covid_deaths = self._prepare_weekly_covid_deaths(region)

            weekly_regional_covid_cases = self._prepare_weekly_covid_cases(region)

            max_deaths = self._prepare_regional_deaths(region)
            print('first max_deaths', max_deaths)
            correction_due_to_aging = self._calculate_correction_due_to_aging(region, max_deaths)
            max_deaths = self._calculate_reference_max_deaths(region, correction_due_to_aging)
            overall_deaths = self._calculate_overall_covid_deaths(region)

            overall_minus_max = np.maximum(0, overall_deaths - max_deaths)
            # corrected_covid_deaths = self._calculate_overall_covid_deaths_with_unreported_deaths(
            #     region, overall_minus_max, weekly_regional_covid_deaths)
            print('daily_regional_deaths', daily_regional_deaths)
            print('weekly_regional_covid_deaths',weekly_regional_covid_deaths)
            print('weekly_regional_covid_cases',weekly_regional_covid_cases)
            print('max_deaths',max_deaths)
            print('correction_due_to_aging',correction_due_to_aging)
            print('overall_deaths',overall_deaths)
            print('overall_minus_max+',overall_minus_max)
            # print(corrected_covid_deaths)
            return overall_minus_max, weekly_regional_covid_deaths
            # self._interpolate_timeline_to_days(corrected_covid_deaths, daily_regional_deaths, ifr, ratios)

    def _prepare_weekly_covid_deaths(self, region: str):
        daily_regional_deaths = self.deaths[self.deaths['location_name'] == region].reset_index(drop=True)
        daily_regional_deaths = daily_regional_deaths.pivot(values='value', index='Age group', columns='date').fillna(0)
        daily_regional_deaths[daily_regional_deaths < 0] = 0


        weekly_regional_deaths = daily_regional_deaths.copy()
        weekly_regional_deaths.columns = [epiweeks.Week.fromdate(datetime.date(year=int(str(c)[:4]),
                                                                               month=int(str(c)[5:7]),
                                                                               day=int(str(c)[8:10]))).isoformat()
                                          for c in daily_regional_deaths.columns]

        weekly_regional_deaths.drop(columns=['2022W01', '2022W02', '2022W03'], inplace=True)
        t = weekly_regional_deaths.transpose()
        weekly_regional_deaths = t.groupby(t.index).sum().transpose()
        return daily_regional_deaths, weekly_regional_deaths

    def _prepare_weekly_covid_cases(self, region: str):
        daily_regional_cases = self.cases[self.cases['location_name'] == region].reset_index(drop=True)
        daily_regional_cases = daily_regional_cases.pivot(values='value', index='Age group', columns='date').fillna(0)[
                               :-1]
        daily_regional_cases[daily_regional_cases < 0] = 0

        weekly_regional_cases = daily_regional_cases.copy()
        weekly_regional_cases.columns = [epiweeks.Week.fromdate(datetime.date(year=int(str(c)[:4]),
                                                                              month=int(str(c)[5:7]),
                                                                              day=int(str(c)[8: 10]))).isoformat()
                                         for c in daily_regional_cases.columns]
        t = weekly_regional_cases.transpose()
        weekly_regional_covid_cases = t.groupby(t.index).sum().transpose()
        return weekly_regional_covid_cases

    def _prepare_regional_deaths(self, region: str):
        if self.all_regions[0] == region:
            max_deaths = pd.DataFrame(self.total_deaths.groupby(['Age group', 'Year']).sum())
            max_deaths = max_deaths.groupby('Age group').apply(max)[max_deaths.columns]
        else:
            max_deaths = self.total_deaths[np.logical_and(self.total_deaths['Region'] == region,
                                                          self.total_deaths['Year'] < 2020)]
            max_deaths = max_deaths.groupby('Age group').apply(max)[max_deaths.columns[3:]]
        return max_deaths

    def _calculate_correction_due_to_aging(self, region: str, max_deaths):
        def group_ages(series, thresholds, names):
            values = dict()
            values['Total'] = series['Total']
            t_prev = 0
            for t, name in zip(thresholds, names):
                values[name] = series[t_prev:t].sum()
                t_prev = t
            values[names[len(thresholds)]] = series[t_prev:-1].sum()
            return {name: values[name] for name in names}

        def calculate_correction(d1, d2):
            return {k: v / d2[k] for k, v in d1.items()}

        age_profile = self.population[region]
        age_profile_old = {k: v[region] for k, v in self.prior_populations.items() if k > 2015}
        age_profile_group = group_ages(age_profile, [65, 75, 85], list(max_deaths.index))
        age_profile_group_old = {k: group_ages(v, [65, 75, 85], list(max_deaths.index)) for k, v in
                                 age_profile_old.items()}
        correction_due_to_aging = {k: calculate_correction(v, age_profile_group) for k, v in
                                   age_profile_group_old.items()}
        return correction_due_to_aging

    def _calculate_reference_max_deaths(self, region, correction_due_to_aging):
        if self.all_regions[0] == region:
            max_deaths_correct = self.total_deaths[self.total_deaths['Year'] < 2020].drop(
                columns=['Region']).reset_index(
                drop=True)
            max_deaths_correct = max_deaths_correct.groupby(['Year', 'Age group']).sum().reset_index()
        else:
            max_deaths_correct = self.total_deaths[
                np.logical_and(self.total_deaths['Region'] == region, self.total_deaths['Year'] < 2020)].drop(
                columns=['Region']).reset_index(drop=True)
        max_deaths_correct['Multiplier'] = max_deaths_correct[['Year', 'Age group']].apply(
            lambda x: correction_due_to_aging[x[0]][x[1]], axis=1)

        for col in range(1, 52 + 1):
            max_deaths_correct[col] = max_deaths_correct[col] / max_deaths_correct['Multiplier']
        max_deaths_correct = max_deaths_correct.groupby('Age group').apply(max)[max_deaths_correct.columns[2:-2]]
        max_deaths_correct.columns = [f'W{i:02d}' for i in range(1, 52 + 1)]
        y2020 = max_deaths_correct.add_prefix('2020')
        y2021 = max_deaths_correct.add_prefix('2021')
        y2020['2020W53'] = np.maximum(max_deaths_correct['W01'], max_deaths_correct['W52'])
        max_deaths_concatenated = pd.concat([y2020, y2021], axis=1)
        return max_deaths_concatenated

    def _calculate_overall_covid_deaths(self, region):
        if self.all_regions[0] == region:
            overall_deaths = self.total_deaths[self.total_deaths['Year'] >= 2020].drop(columns=['Region']).reset_index(
                drop=True)
            overall_deaths = overall_deaths.groupby(['Year', 'Age group']).sum().reset_index()
            y2020 = overall_deaths[:5].reset_index(drop=True)[overall_deaths.columns[1:]]
            y2020.columns = ['Age group'] + [f'2020W{i:02d}' for i in range(1, 53 + 1)]
            y2021 = overall_deaths[-5:].reset_index(drop=True)[overall_deaths.columns[2:-1]]
            y2021.columns = [f'2021W{i:02d}' for i in range(1, 52 + 1)]
            overall_deaths = pd.concat([y2020[y2020.columns[1:]], y2021], axis=1)
            overall_deaths.index = y2020['Age group']

        else:
            overall_deaths = self.total_deaths[
                np.logical_and(self.total_deaths['Region'] == region, self.total_deaths['Year'] >= 2020)].drop(
                columns=['Region']).reset_index(drop=True)
            y2020 = overall_deaths[-5:].reset_index(drop=True)[overall_deaths.columns[1:]]
            y2020.columns = ['Age group'] + [f'2020W{i:02d}' for i in range(1, 53 + 1)]
            y2021 = overall_deaths[:5].reset_index(drop=True)[overall_deaths.columns[2:-1]]
            y2021.columns = [f'2021W{i:02d}' for i in range(1, 52 + 1)]
            overall_deaths = pd.concat([y2020[y2020.columns[1:]], y2021], axis=1)
            overall_deaths.index = y2020['Age group']
        return overall_deaths

    def _calculate_overall_covid_deaths_with_unreported_deaths(self, region, overall_minus_max,
                                                               weekly_regional_covid_deaths):
        def group_ages(series, thresholds, names):
            values = dict()
            values['Total'] = series['Total']
            t_prev = 0
            for t, name in zip(thresholds, names):
                values[name] = series[t_prev:t].sum()
                t_prev = t
            values[names[len(thresholds)]] = series[t_prev:-1].sum()
            return {name: values[name] for name in names}

        age_profile = self.population[region]
        ''' '''
        keys_in_covid_deaths = self.deaths['Age group'].unique()
        keys_in_death = [x for x in self.total_deaths['Age group'].unique() if x != 'Total']
        keys_1 = [int(key.split('-')[0]) if len(key.split('-')) == 2 else int(key.split('+')[0]) for key in
                  keys_in_covid_deaths]
        keys_2 = [int(key.split('-')[0]) if len(key.split('-')) == 2 else int(key.split('+')[0]) for key in
                  keys_in_death]
        needed_keys = sorted({*keys_1, *keys_2})

        keys = [f'{low:02d}-{high:02d}' if high != '+' else f'{low}{high}' for low, high in
                zip(needed_keys, needed_keys[1:] + ['+'])]

        dictionary = {}
        for low, high in zip(keys_1, keys_1[1:] + ['+']):
            dictionary[f'{low:02d}-{high - 1:02d}' if high != '+' else f'{low}+'] = []
            for low_2, high_2 in zip(keys_2, keys_2[1:] + ['+']):
                if high == '+':
                    if low <= low_2:
                        dictionary[f'{low:02d}+'] += [f'{low_2:02d}+']
                    if high_2 != '+' and high_2 >= low:
                        dictionary[f'{low:02d}+'] += [f'{low_2:02d}-{high_2:02d}']
                elif high_2 != '+' and high != '+':
                    if low_2 <= low <= high_2 and low_2 <= high <= high_2:
                        dictionary[f'{low:02d}-{high - 1:02d}'] += [f'{low_2:02d}-{high_2:02d}']
                    if low <= high_2 <= high:
                        dictionary[f'{low:02d}-{high - 1:02d}'] += [f'{low_2:02d}-{high_2:02d}']
                    if low <= low_2 <= high:
                        dictionary[f'{low:02d}-{high - 1:02d}'] += [f'{low_2:02d}-{high_2:02d}']

        dictionary = {key_2: set(value) for key_2, value in dictionary.items()}
        dictionary_2 = {}
        for low, high in zip(keys_1, keys_1[1:] + [max(needed_keys)]):
            if needed_keys.index(high) == len(needed_keys) - 1:
                dictionary_2[f'{low:02d}+'] = {i for i in keys[needed_keys.index(low):]}
            else:
                dictionary_2[f'{low:02d}-{high - 1:02d}'] = {i for i in
                                                             keys[needed_keys.index(low): needed_keys.index(high)]}
        result = {key: {key_2: value for key_2, value in zip(sorted(dictionary[key]), sorted(dictionary_2[key]))} for
                  key in dictionary.keys()}
        ''''''
        keys_3 = [int(key.split('-')[0]) if len(key.split('-')) == 2 else int(key.split('+')[0]) for key in
                  list(weekly_regional_covid_deaths.index)]
        self.age_profile_regional_group2_deaths_age_groups = group_ages(age_profile, keys_2[1:], keys_in_death)
        self.age_profile_regional_group2_fix_age_groups = group_ages(age_profile, needed_keys[1:], keys)
        self.age_profile_regional_group2_covid_age_groups = group_ages(age_profile, keys_3[1:],
                                                                       list(weekly_regional_covid_deaths.index))
        fix_age_groups = {key: {
            key_2: self.age_profile_regional_group2_fix_age_groups[value_2] /
                   self.age_profile_regional_group2_deaths_age_groups[
                       key_2] for key_2, value_2 in result[key].items()} for key in result.keys()}

        def apply_age_fix_mapping(obj, d):
            s = 0
            for k, v in d.items():
                s += obj[k] * v
            return s

        series = []
        for k, v in fix_age_groups.items():
            series.append(overall_minus_max.apply(lambda o: apply_age_fix_mapping(o, v), axis=0))
        overall_minus_max_reorg = pd.concat(series, axis=1).transpose()
        overall_minus_max_reorg.index = fix_age_groups.keys()
        additional_deaths_not_attributed_to_covid = np.maximum(
            0, overall_minus_max_reorg[weekly_regional_covid_deaths.columns] - weekly_regional_covid_deaths)
        corrected_covid_deaths = (additional_deaths_not_attributed_to_covid * 0.85 + weekly_regional_covid_deaths)
        return corrected_covid_deaths

    def _interpolate_timeline_to_days(self, corrected_covid_deaths, daily_regional_deaths, ifr, ratios=None):
        filter_cols = []
        corrected_covid_deaths_regional_daily = corrected_covid_deaths.copy()
        for week in corrected_covid_deaths.columns:
            epi = epiweeks.Week.fromstring(week)
            for day in epi.iterdates():
                filter_cols.append(day)
                corrected_covid_deaths_regional_daily[day] = corrected_covid_deaths[week] / 7
        corrected_covid_deaths_regional_daily = corrected_covid_deaths_regional_daily[filter_cols]
        '''
        Plot Covid deaths cumulative sum 
        '''
        plt.figure()
        plt.plot(corrected_covid_deaths_regional_daily.transpose().cumsum())
        plt.title('Covid death cumulative sum')
        plt.xlabel('day')
        plt.ylabel('Number of deaths')
        plt.legend(corrected_covid_deaths_regional_daily.index)
        plt.savefig(self.result_dir + 'covid deaths cumulative sum.png')

        # apply delay from detection to death
        if ratios is not None:
            detections_time_of_corrected_covid_deaths_regional_daily = corrected_covid_deaths_regional_daily.copy()
            for i, row in detections_time_of_corrected_covid_deaths_regional_daily.iterrows():
                detections_time_of_corrected_covid_deaths_regional_daily.loc[i] = np.correlate(row.to_numpy(), ratios,
                                                                                               'full')[79:-4]
        else:
            detections_time_of_corrected_covid_deaths_regional_daily = corrected_covid_deaths_regional_daily.copy()

        daily_regional_deaths.columns = [datetime.datetime.strptime(col, '%Y-%m-%d').date() for col in
                                         daily_regional_deaths.columns]

        previously_infected = detections_time_of_corrected_covid_deaths_regional_daily.copy()
        cases_regional_filtered = daily_regional_deaths
        to_calculation = None
        for key, value in ifr.items():
            to_calculation = list(value.keys())
        for condition in previously_infected.index:
            if condition not in to_calculation:
                previously_infected.drop(previously_infected.loc[previously_infected.index == condition].index,
                                         inplace=True)
                cases_regional_filtered.drop(cases_regional_filtered.loc[cases_regional_filtered.index == condition].index,
                                             inplace=True)
        cases_regional_filtered = cases_regional_filtered.transpose()
        previously_infected = previously_infected.transpose()
        previously_infected = previously_infected.transpose()[
            cases_regional_filtered[cases_regional_filtered.columns[0]].index].transpose()

        fig, axes = plt.subplots(5, 3, figsize=(17, 23))
        for j, (k, ifr_details) in enumerate(ifr.items()):
            ax = axes[j]
            for i, col in enumerate(previously_infected.columns):
                (previously_infected[col] / ifr_details[col][1] / self.age_profile_regional_group2_covid_age_groups[
                    col]).cumsum().plot(ax=ax[i], label=f'{col} estimate')
                (previously_infected[col] / ifr_details[col][0] / self.age_profile_regional_group2_covid_age_groups[
                    col]).cumsum().plot(ax=ax[i], label=f'{col} upper bound (95% conf. interv.)')
                (previously_infected[col] / ifr_details[col][2] / self.age_profile_regional_group2_covid_age_groups[
                    col]).cumsum().plot(ax=ax[i], label=f'{col} lower bound (95% conf. interv.)')
                (cases_regional_filtered[col] / self.age_profile_regional_group2_covid_age_groups[col]).cumsum().plot(
                    ax=ax[i],
                    label=f'{col} detected cases')
                ax[i].legend(loc='upper left')
                ax[i].grid()
                ax[i].set_ylabel('fraction of age group population')
                ax[i].set_title(f'IFR by {k} (age group {col})')
                fig.autofmt_xdate()
                ax[i].set_ylim([0, None])
        plt.savefig(self.result_dir + f'fraction of previously infected.png')

        fig, axes = plt.subplots(5, 3, figsize=(17, 23))
        for j, (k, ifr_details) in enumerate(ifr.items()):
            ax = axes[j]
            for i, col in enumerate(previously_infected.columns):
                (previously_infected[col] / ifr_details[col][1] / self.age_profile_regional_group2_covid_age_groups[
                    col]).plot(
                    ax=ax[i], label=f'{col} estimate')
                (previously_infected[col] / ifr_details[col][0] / self.age_profile_regional_group2_covid_age_groups[
                    col]).plot(
                    ax=ax[i], label=f'{col} upper bound (95% conf. interv.)')
                (previously_infected[col] / ifr_details[col][2] / self.age_profile_regional_group2_covid_age_groups[
                    col]).plot(
                    ax=ax[i], label=f'{col} lower bound (95% conf. interv.)')
                (cases_regional_filtered[col] / self.age_profile_regional_group2_covid_age_groups[col]).rolling(
                    7).mean().plot(
                    ax=ax[i], label=f'{col} detected cases')
                ax[i].legend(loc='upper left')
                ax[i].grid()
                ax[i].set_ylabel('Fraction of age group population')
                ax[i].set_title(f'IFR by {k} (age group {col})')
                fig.autofmt_xdate()
                ax[i].set_ylim([0, None])
        plt.savefig(self.result_dir + f'fraction of previously infected timeline.png')

        fig, axes = plt.subplots(5, 3, figsize=(17, 23))
        for j, (k, ifr_details) in enumerate(ifr.items()):
            ax = axes[j]
            for i, col in enumerate(previously_infected.columns):
                (previously_infected[col] / ifr_details[col][1]).cumsum().plot(ax=ax[i], label=f'{col} estimate')
                (previously_infected[col] / ifr_details[col][0]).cumsum().plot(ax=ax[i],
                                                                               label=f'{col} upper bound (95% conf. interv.)')
                (previously_infected[col] / ifr_details[col][2]).cumsum().plot(ax=ax[i],
                                                                               label=f'{col} lower bound (95% conf. interv.)')
                (cases_regional_filtered[col]).cumsum().plot(ax=ax[i], label=f'{col} detected cases')
                ax[i].legend(loc='upper left')
                ax[i].grid()
                ax[i].set_ylabel('Number of people')
                ax[i].set_title(f'IFR by {k} (age group {col})')
                fig.autofmt_xdate()
                ax[i].set_ylim([0, None])
        plt.savefig(self.result_dir + f'number of previously infected.png')

        fig, axes = plt.subplots(5, 3, figsize=(17, 23))
        for j, (k, ifr_details) in enumerate(ifr.items()):
            ax = axes[j]
            for i, col in enumerate(previously_infected.columns):
                (previously_infected[col] / ifr_details[col][1]).plot(ax=ax[i], label=f'{col} estimate')
                (previously_infected[col] / ifr_details[col][0]).plot(ax=ax[i],
                                                                      label=f'{col} upper bound (95% conf. interv.)')
                (previously_infected[col] / ifr_details[col][2]).plot(ax=ax[i],
                                                                      label=f'{col} lower bound (95% conf. interv.)')
                (cases_regional_filtered[col]).rolling(7).mean().plot(ax=ax[i], label=f'{col} detected cases')
                ax[i].legend(loc='upper left')
                ax[i].grid()
                ax[i].set_ylabel('Number of people')
                ax[i].set_title(f'IFR by {k} (age group {col})')
                fig.autofmt_xdate()
                ax[i].set_ylim([0, None])
        plt.savefig(self.result_dir + f'number of previously infected timeline.png')

        fig, axes = plt.subplots(5, 3, figsize=(17, 23))
        for j, (k, ifr_details) in enumerate(ifr.items()):
            ax = axes[j]
            for i, col in enumerate(previously_infected.columns):
                (previously_infected[col] / ifr_details[col][1] / (cases_regional_filtered[col]).rolling(
                    7).mean()).plot(
                    ax=ax[i], label=f'{col} estimate')
                (previously_infected[col] / ifr_details[col][0] / (cases_regional_filtered[col]).rolling(
                    7).mean()).plot(
                    ax=ax[i], label=f'{col} upper bound (95% conf. interv.)')
                (previously_infected[col] / ifr_details[col][2] / (cases_regional_filtered[col]).rolling(
                    7).mean()).plot(
                    ax=ax[i], label=f'{col} lower bound (95% conf. interv.)')
                ax[i].legend(loc='upper left')
                ax[i].grid()
                ax[i].set_ylabel('Dark figure')
                ax[i].set_title(f'IFR by {k} (age group {col})')
                fig.autofmt_xdate()
                ax[i].set_ylim([0, None])
        plt.savefig(self.result_dir + f'darkfigure of previously infected timeline.png')

        fig, axes = plt.subplots(5, 3, figsize=(17, 23))
        for j, (k, ifr_details) in enumerate(ifr.items()):
            ax = axes[j]
            for i, col in enumerate(previously_infected.columns):
                ((previously_infected[col] / ifr_details[col][1]).cumsum() / (
                    cases_regional_filtered[col].cumsum())).plot(ax=ax[i], label=f'{col} estimate')
                ((previously_infected[col] / ifr_details[col][0]).cumsum() / (
                    cases_regional_filtered[col].cumsum())).plot(ax=ax[i],
                                                                 label=f'{col} upper bound (95% conf. interv.)')
                ((previously_infected[col] / ifr_details[col][2]).cumsum() / (
                    cases_regional_filtered[col].cumsum())).plot(ax=ax[i],
                                                                 label=f'{col} lower bound (95% conf. interv.)')
                ax[i].legend(loc='upper left')
                ax[i].grid()
                ax[i].set_ylabel('Dark figure')
                ax[i].set_title(f'IFR by {k} (age group {col})')
                fig.autofmt_xdate()
                ax[i].set_ylim([0, None])
        plt.savefig(self.result_dir + f'cumulative darkfigure of previously infected.png')

#  TODO: make it work (calculation IFR)

# IFRs = {
#     'O\'Driscoll': {
#         'A35-A59': [0.122, 0.115, 0.128],
#         'A60-A79': [0.992, 0.942, 1.045],
#         'A80+': [7.274, 6.909, 7.656]
#     },
#     'Verity': {
#         'A35-A59': [0.349, 0.194, 0.743],
#         'A60-A79': [2.913, 1.670, 5.793],
#         'A80+': [7.800, 3.800, 13.30]
#     },
#     'Perez-Saez': {
#         'A35-A59': [0.070, 0.047, 0.097],
#         'A60-A79': [3.892, 2.985, 5.145],
#         'A80+': [5.600, 4.300, 7.400]
#     },
#     'Levin': {
#         'A35-A59': [0.226, 0.212, 0.276],
#         'A60-A79': [2.491, 2.294, 3.266],
#         'A80+': [15.61, 12.20, 19.50]
#     }
# }
# IFRs = {k: {k1: sorted(np.array(v1) / 100) for k1, v1 in v.items()} for k, v in IFRs.items()}
# ifr = pd.read_excel('./data/IFR_estimation.xlsx')
# ifr.index = ifr['Age group']
# ifr = ifr[ifr.columns[1:]]
# age_profile_regional_group3_fix_age_groups = group_ages(age_profile,
#                                                         [35, 40, 45, 50, 55, 60, 65, 70, 75, 80],
#                                                         ['0-35', '35-39', '40-44', '45-49', '50-54', '55-59',
#                                                          '60-64', '65-69', '70-74', '75-79', '80+'])
#
# grouped_age_groups = {
#     '35-59': ['35-39', '40-44', '45-49', '50-54', '55-59'],
#     '60-79': ['60-64', '65-69', '70-74', '75-79'],
#     '80+': ['80+']
# }
# weekly_regional_covid_deaths = self._prepare_weekly_covid_cases(region)
# print(weekly_regional_covid_deaths)
# age_profile_regional_group2_covid_age_groups = group_ages(age_profile, [5, 15, 35, 60, 80],
#                                                           list(weekly_regional_covid_deaths.index))
# IFRs['Driscoll (ours)'] = dict()
# for k, vs in grouped_age_groups.items():
#     print(k)
#     grouped_sum = np.zeros(3)
#     for elem in vs:
#         grouped_sum += ifr.loc[elem].to_numpy() * age_profile_regional_group3_fix_age_groups[elem] / \
#                        age_profile_regional_group2_covid_age_groups[k]
#     IFRs['Driscoll (ours)'][k] = list(grouped_sum)
#
# print(IFRs['Driscoll (ours)'])
