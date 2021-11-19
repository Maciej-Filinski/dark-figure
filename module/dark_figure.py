import pandas as pd
import numpy as np
import os
import datetime
import epiweeks


"""
what we need:

cases ->
deaths ->
population ->
ifr ->
test to deaths ration ->
"""


class DarkFigure:
    def __init__(self, deaths: pd.DataFrame, cases: pd.DataFrame, total_deaths: pd.DataFrame, population, prior_populations):
        self.deaths = deaths
        self.cases = cases
        self.total_deaths = total_deaths
        self.population = population
        self.prior_populations = prior_populations
        self.all_regions = None

    def __call__(self, all_regions: list):
        self.all_regions = all_regions
        for region in all_regions:
            weekly_regional_covid_deaths = self._prepare_weekly_covid_deaths(region)
            weekly_regional_covid_cases = self._prepare_weekly_covid_cases(region)
            max_deaths = self._prepare_regional_deaths(region)
            correction_due_to_aging = self._calculate_correction_due_to_aging(region, max_deaths)
            max_deaths = self._calculate_reference_max_deaths(region, correction_due_to_aging)
            overall_deaths = self._calculate_overall_covid_deaths(region)

            overall_minus_max = np.maximum(0, overall_deaths - max_deaths)
            corrected_covid_deaths = self._calculate_overall_covid_deaths_with_unreported_deaths(
                region, overall_minus_max, weekly_regional_covid_deaths)
            self._interpolate_timeline_to_days(corrected_covid_deaths)

    def _prepare_weekly_covid_deaths(self, region: str):
        daily_regional_deaths = self.deaths[self.deaths['location_name'] == region].reset_index(drop=True)
        daily_regional_deaths = daily_regional_deaths.pivot(values='value', index='age_group', columns='date').fillna(0)
        daily_regional_deaths[daily_regional_deaths < 0] = 0

        weekly_regional_deaths = daily_regional_deaths.copy()
        weekly_regional_deaths.columns = [epiweeks.Week.fromdate(datetime.date(year=int(c[:4]),
                                                                               month=int(c[5:7]),
                                                                               day=int(c[8:10]))).isoformat()
                                          for c in daily_regional_deaths.columns]
        t = weekly_regional_deaths.transpose()
        weekly_regional_deaths = t.groupby(t.index).sum().transpose()
        return weekly_regional_deaths

    def _prepare_weekly_covid_cases(self, region: str):
        daily_regional_cases = self.cases[self.cases['location_name'] == region].reset_index(drop=True)
        daily_regional_cases = daily_regional_cases.pivot(values='value', index='age_group', columns='date').fillna(0)[:-1]
        daily_regional_cases[daily_regional_cases < 0] = 0

        weekly_regional_cases = daily_regional_cases.copy()
        weekly_regional_cases.columns = [epiweeks.Week.fromdate(datetime.date(year=int(c[:4]),
                                                                              month=int(c[5:7]),
                                                                              day=int(c[8: 10]))).isoformat()
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
            max_deaths_correct = self.total_deaths[self.total_deaths['Year'] < 2020].drop(columns=['Region']).reset_index(
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

    def _calculate_overall_covid_deaths_with_unreported_deaths(self, region, overall_minus_max, weekly_regional_covid_deaths):
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

        age_profile_regional_group2_deaths_age_groups = group_ages(age_profile, [65, 75, 85],
                                                                 list(overall_minus_max.index))
        age_profile_regional_group2_fix_age_groups = group_ages(age_profile, [5, 15, 35, 60, 65, 75, 80, 85],
                                                              ['0-5', '5-15', '15-35', '35-60', '60-65',
                                                               '65-75', '75-80', '80-85', '85+'])
        fix_age_groups = {
            'A00-A04': {
                '0-65': age_profile_regional_group2_fix_age_groups['0-5'] / age_profile_regional_group2_deaths_age_groups[
                    '0-65']
            },
            'A05-A14': {
                '0-65': age_profile_regional_group2_fix_age_groups['5-15'] / age_profile_regional_group2_deaths_age_groups[
                    '0-65']
            },
            'A15-A34': {
                '0-65': age_profile_regional_group2_fix_age_groups['15-35'] / age_profile_regional_group2_deaths_age_groups[
                    '0-65']
            },
            'A35-A59': {
                '0-65': age_profile_regional_group2_fix_age_groups['35-60'] / age_profile_regional_group2_deaths_age_groups[
                    '0-65']
            },
            'A60-A79': {
                '0-65': age_profile_regional_group2_fix_age_groups['60-65'] / age_profile_regional_group2_deaths_age_groups[
                    '0-65'],
                '65-75': 1,
                '75-85': age_profile_regional_group2_fix_age_groups['75-80'] /
                         age_profile_regional_group2_deaths_age_groups['75-85']
            },
            'A80+': {
                '75-85': age_profile_regional_group2_fix_age_groups['80-85'] /
                         age_profile_regional_group2_deaths_age_groups['75-85'],
                '85+': 1
            }
        }
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

    def _interpolate_timeline_to_days(self, corrected_covid_deaths):
        filter_cols = []
        corrected_covid_deaths_regional_daily = corrected_covid_deaths.copy()
        for week in corrected_covid_deaths.columns:
            epi = epiweeks.Week.fromstring(week)
            for day in epi.iterdates():
                filter_cols.append(day)
                corrected_covid_deaths_regional_daily[day] = corrected_covid_deaths[week] / 7
        corrected_covid_deaths_regional_daily = corrected_covid_deaths_regional_daily[filter_cols]
        corrected_covid_deaths_regional_daily.transpose().cumsum().plot()

        # apply delay from detection to death
        detections_time_of_corrected_covid_deaths_regional_daily = corrected_covid_deaths_regional_daily.copy()
        ratios = pd.read_csv('./data/positive_test_to_death_days_distribution.csv')
        ratios.index = ratios['offset']
        ratios = ratios['probs']
        correlation_type = 'full'
        for i, row in detections_time_of_corrected_covid_deaths_regional_daily.iterrows():
            detections_time_of_corrected_covid_deaths_regional_daily.loc[i] = np.correlate(row.to_numpy(), ratios,
                                                                                           correlation_type)[79:-4]

        detections_time_of_corrected_covid_deaths_regional_daily.transpose().cumsum().plot()
    
    # def plot(self, region):
    #     RESULT_DIR = f'./result/{str(datetime.datetime.now().date())}/'
    #     if not os.path.exists(RESULT_DIR):
    #         os.makedirs(RESULT_DIR)
    #
    #     IFRs = {
    #         'O\'Driscoll': {
    #             'A35-A59': [0.122, 0.115, 0.128],
    #             'A60-A79': [0.992, 0.942, 1.045],
    #             'A80+': [7.274, 6.909, 7.656]
    #         },
    #         'Verity': {
    #             'A35-A59': [0.349, 0.194, 0.743],
    #             'A60-A79': [2.913, 1.670, 5.793],
    #             'A80+': [7.800, 3.800, 13.30]
    #         },
    #         'Perez-Saez': {
    #             'A35-A59': [0.070, 0.047, 0.097],
    #             'A60-A79': [3.892, 2.985, 5.145],
    #             'A80+': [5.600, 4.300, 7.400]
    #         },
    #         'Levin': {
    #             'A35-A59': [0.226, 0.212, 0.276],
    #             'A60-A79': [2.491, 2.294, 3.266],
    #             'A80+': [15.61, 12.20, 19.50]
    #         }
    #     }
    #     IFRs = {k: {k1: sorted(np.array(v1) / 100) for k1, v1 in v.items()} for k, v in IFRs.items()}
    #
    #     ifr = pd.read_excel('./data/Germany/IFR_estimation.xlsx')
    #     ifr.index = ifr['Age group']
    #     ifr = ifr[ifr.columns[1:]]
    #
    #     age_profile_regional_group3_fix_age_groups = group_ages(age_profile_regional,
    #                                                           [35, 40, 45, 50, 55, 60, 65, 70, 75, 80],
    #                                                           ['0-35', '35-39', '40-44', '45-49', '50-54', '55-59',
    #                                                            '60-64', '65-69', '70-74', '75-79', '80+'])
    #
    #     grouped_age_groups = {
    #         'A35-A59': ['35-39', '40-44', '45-49', '50-54', '55-59'],
    #         'A60-A79': ['60-64', '65-69', '70-74', '75-79'],
    #         'A80+': ['80+']
    #     }
    #     IFRs['Driscoll (ours)'] = dict()
    #     for k, vs in grouped_age_groups.items():
    #         print(k)
    #         grouped_sum = np.zeros(3)
    #         for elem in vs:
    #             grouped_sum += ifr.loc[elem].to_numpy() * age_profile_regional_group3_fix_age_groups[elem] / \
    #                            age_profile_regional_group2_covid_age_groups[k]
    #         IFRs['Driscoll (ours)'][k] = list(grouped_sum)
    #
    #     print(IFRs['Driscoll (ours)'])
    #
    #     previously_infected = detections_time_of_corrected_covid_deaths_regional_daily[4:].transpose()
    #     previously_infected = previously_infected[['A60-A79', 'A80+']]
    #     print('previously_infected', previously_infected)
    #
    #     cases_regional.columns = [datetime.datetime.strptime(col, '%Y-%m-%d').date() for col in cases_regional.columns]
    #
    #     cases_regional_filtered = cases_regional[4:].transpose()
    #     print(cases_regional_filtered.columns)
    #     print('previously infected transposed', previously_infected.transpose())
    #     previously_infected = previously_infected.transpose()[
    #         cases_regional_filtered[cases_regional_filtered.columns[0]].index].transpose()
    #
    #     from matplotlib import pyplot as plt
    #     fig, axes = plt.subplots(5, 2, figsize=(17, 23))
    #     for j, (k, ifr_details) in enumerate(IFRs.items()):
    #         ax = axes[j]
    #         for i, col in enumerate(previously_infected.columns):
    #             (previously_infected[col] / ifr_details[col][1] / age_profile_regional_group2_covid_age_groups[
    #                 col]).cumsum().plot(ax=ax[i], label=f'{col} estimate')
    #             (previously_infected[col] / ifr_details[col][0] / age_profile_regional_group2_covid_age_groups[
    #                 col]).cumsum().plot(ax=ax[i], label=f'{col} upper bound (95% conf. interv.)')
    #             (previously_infected[col] / ifr_details[col][2] / age_profile_regional_group2_covid_age_groups[
    #                 col]).cumsum().plot(ax=ax[i], label=f'{col} lower bound (95% conf. interv.)')
    #             (cases_regional_filtered[col] / age_profile_regional_group2_covid_age_groups[col]).cumsum().plot(ax=ax[i],
    #                                                                                                          label=f'{col} detected cases')
    #             ax[i].legend(loc='upper left')
    #             ax[i].grid()
    #             ax[i].set_ylabel('fraction of age group population')
    #             ax[i].set_title(f'{region} - IFR by {k} (age group {col})')
    #             fig.autofmt_xdate()
    #             ax[i].set_ylim([0, None])
    #     plt.savefig(RESULT_DIR + f'{region}_20211025_fraction_of_previously_infected_60+.png')
    #
    #     from matplotlib import pyplot as plt
    #     fig, axes = plt.subplots(5, 2, figsize=(17, 23))
    #     for j, (k, ifr_details) in enumerate(IFRs.items()):
    #         ax = axes[j]
    #         for i, col in enumerate(previously_infected.columns):
    #             (previously_infected[col] / ifr_details[col][1] / age_profile_regional_group2_covid_age_groups[col]).plot(
    #                 ax=ax[i], label=f'{col} estimate')
    #             (previously_infected[col] / ifr_details[col][0] / age_profile_regional_group2_covid_age_groups[col]).plot(
    #                 ax=ax[i], label=f'{col} upper bound (95% conf. interv.)')
    #             (previously_infected[col] / ifr_details[col][2] / age_profile_regional_group2_covid_age_groups[col]).plot(
    #                 ax=ax[i], label=f'{col} lower bound (95% conf. interv.)')
    #             (cases_regional_filtered[col] / age_profile_regional_group2_covid_age_groups[col]).rolling(7).mean().plot(
    #                 ax=ax[i], label=f'{col} detected cases')
    #             ax[i].legend(loc='upper left')
    #             ax[i].grid()
    #             ax[i].set_ylabel('Fraction of age group population')
    #             ax[i].set_title(f'{region} - IFR by {k} (age group {col})')
    #             fig.autofmt_xdate()
    #             ax[i].set_ylim([0, None])
    #     plt.savefig(RESULT_DIR + f'{region}_20211025_fraction_of_previously_infected_60+_timeline.png')
    #
    #     from matplotlib import pyplot as plt
    #     fig, axes = plt.subplots(5, 2, figsize=(17, 23))
    #     for j, (k, ifr_details) in enumerate(IFRs.items()):
    #         ax = axes[j]
    #         for i, col in enumerate(previously_infected.columns):
    #             (previously_infected[col] / ifr_details[col][1]).cumsum().plot(ax=ax[i], label=f'{col} estimate')
    #             (previously_infected[col] / ifr_details[col][0]).cumsum().plot(ax=ax[i],
    #                                                                            label=f'{col} upper bound (95% conf. interv.)')
    #             (previously_infected[col] / ifr_details[col][2]).cumsum().plot(ax=ax[i],
    #                                                                            label=f'{col} lower bound (95% conf. interv.)')
    #             (cases_regional_filtered[col]).cumsum().plot(ax=ax[i], label=f'{col} detected cases')
    #             ax[i].legend(loc='upper left')
    #             ax[i].grid()
    #             ax[i].set_ylabel('Number of people')
    #             ax[i].set_title(f'{region} - IFR by {k} (age group {col})')
    #             fig.autofmt_xdate()
    #             ax[i].set_ylim([0, None])
    #     plt.savefig(RESULT_DIR + f'{region}_20211025_number_of_previously_infected_60+.png')
    #
    #     from matplotlib import pyplot as plt
    #     fig, axes = plt.subplots(5, 2, figsize=(17, 23))
    #     for j, (k, ifr_details) in enumerate(IFRs.items()):
    #         ax = axes[j]
    #         for i, col in enumerate(previously_infected.columns):
    #             (previously_infected[col] / ifr_details[col][1]).plot(ax=ax[i], label=f'{col} estimate')
    #             (previously_infected[col] / ifr_details[col][0]).plot(ax=ax[i],
    #                                                                   label=f'{col} upper bound (95% conf. interv.)')
    #             (previously_infected[col] / ifr_details[col][2]).plot(ax=ax[i],
    #                                                                   label=f'{col} lower bound (95% conf. interv.)')
    #             (cases_regional_filtered[col]).rolling(7).mean().plot(ax=ax[i], label=f'{col} detected cases')
    #             ax[i].legend(loc='upper left')
    #             ax[i].grid()
    #             ax[i].set_ylabel('Number of people')
    #             ax[i].set_title(f'{region} - IFR by {k} (age group {col})')
    #             fig.autofmt_xdate()
    #             ax[i].set_ylim([0, None])
    #     plt.savefig(RESULT_DIR + f'{region}_20211025_number_of_previously_infected_timeline_60+.png')
    #
    #     from matplotlib import pyplot as plt
    #     fig, axes = plt.subplots(5, 2, figsize=(17, 23))
    #     for j, (k, ifr_details) in enumerate(IFRs.items()):
    #         ax = axes[j]
    #         for i, col in enumerate(previously_infected.columns):
    #             (previously_infected[col] / ifr_details[col][1] / (cases_regional_filtered[col]).rolling(7).mean()).plot(
    #                 ax=ax[i], label=f'{col} estimate')
    #             (previously_infected[col] / ifr_details[col][0] / (cases_regional_filtered[col]).rolling(7).mean()).plot(
    #                 ax=ax[i], label=f'{col} upper bound (95% conf. interv.)')
    #             (previously_infected[col] / ifr_details[col][2] / (cases_regional_filtered[col]).rolling(7).mean()).plot(
    #                 ax=ax[i], label=f'{col} lower bound (95% conf. interv.)')
    #             ax[i].legend(loc='upper left')
    #             ax[i].grid()
    #             ax[i].set_ylabel('Dark figure')
    #             ax[i].set_title(f'{region} - IFR by {k} (age group {col})')
    #             fig.autofmt_xdate()
    #             ax[i].set_ylim([0, None])
    #     plt.savefig(RESULT_DIR + f'{region}_20211025_darkfigure_of_previously_infected_timeline_60+.png')
    #
    #     from matplotlib import pyplot as plt
    #     fig, axes = plt.subplots(5, 2, figsize=(17, 23))
    #     for j, (k, ifr_details) in enumerate(IFRs.items()):
    #         ax = axes[j]
    #         for i, col in enumerate(previously_infected.columns):
    #             ((previously_infected[col] / ifr_details[col][1]).cumsum() / (
    #                 cases_regional_filtered[col].cumsum())).plot(ax=ax[i], label=f'{col} estimate')
    #             ((previously_infected[col] / ifr_details[col][0]).cumsum() / (
    #                 cases_regional_filtered[col].cumsum())).plot(ax=ax[i], label=f'{col} upper bound (95% conf. interv.)')
    #             ((previously_infected[col] / ifr_details[col][2]).cumsum() / (
    #                 cases_regional_filtered[col].cumsum())).plot(ax=ax[i], label=f'{col} lower bound (95% conf. interv.)')
    #             ax[i].legend(loc='upper left')
    #             ax[i].grid()
    #             ax[i].set_ylabel('Dark figure')
    #             ax[i].set_title(f'{region} - IFR by {k} (age group {col})')
    #             fig.autofmt_xdate()
    #             ax[i].set_ylim([0, None])
    #     plt.savefig(RESULT_DIR + f'{region}_20211025_cumulative_darkfigure_of_previously_infected_60+.png')