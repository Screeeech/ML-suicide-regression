import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import mlab

import learning

allData = pd.read_csv("master.csv")


def get_country(country):
    country_data = allData.loc[allData.country == country]
    country_data.drop(['country-year', 'HDI for year', 'gdp_for_year ($)', 'gdp_per_capita ($)', 'generation'],
                      axis=1, inplace=True)

    country_data_len = len(country_data)
    first_index = country_data.head(1).index.values.astype(int)[0]
    last_index = first_index + country_data_len - 1

    first_year = country_data.loc[first_index]['year']
    last_year = country_data.loc[last_index]['year']

    total_death = []
    total_death_index = []

    for i in range(first_year, last_year + 1):
        try:
            total_death.append(country_data.groupby('year').get_group(i)['suicides/100k pop'].sum())
            total_death_index.append(i)
        except:
            pass

    return pd.Series(total_death, index=total_death_index, name=country)


def find_common_min(*country_series):
    common_min_year = 0
    for country in country_series:
        first_year = country.head(1).index.values.astype(int)[0]

        if first_year > common_min_year:
            common_min_year = first_year

    return common_min_year


def find_common_max(*countrySeries):
    common_max_year = 2021
    for country in countrySeries:
        first_year = country.tail(1).index.values.astype(int)[0]

        if first_year < common_max_year:
            common_max_year = first_year

    return common_max_year


country = get_country('United States')
common_year_index = []
year_deaths = []

for i in range(find_common_min(country), 2002):
    common_year_index.append(i - find_common_min(country))
    # x axis as years since 1985
    try:
        year_deaths.append(np.round(country.loc[i] / 10, decimals=2))
        # y axis tens of deaths
    except:
        pass

z = learning.Learn(common_year_index, year_deaths, 0, 0, 1)

# contour plot
zoom = 1
fig = plt.figure()
ax = fig.add_subplot(111)
x_space = np.linspace(-75 * zoom, 110 * zoom, 1000)
y_space = np.linspace(-10 * zoom, 10 * zoom, 1000)
x, y = np.meshgrid(x_space, y_space)
ax.contour(x, y, z.get_cost(x, y), levels=np.arange(-16000, 16000, 300))

z.theta_0 = 75
z.theta_1 = -7.5
z.alpha = 0.01
theta_0 = []
theta_1 = []
for i in range(5000):
    theta_0.append(z.theta_0)
    theta_1.append(z.theta_1)
    z.adjust_by_gradient()
    i += 1
print(theta_0[len(theta_0) - 1])
print(theta_1[len(theta_1) - 1])
plt.scatter(theta_0, theta_1)

plt.show()


# scatter plot
x = np.linspace(0, 20, 100)
y = theta_1[len(theta_1) - 1] * x + theta_0[len(theta_0) - 1]
plt.plot(x, y)
plt.scatter(common_year_index, year_deaths)
plt.show()



