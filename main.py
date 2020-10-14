import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import learning
import test

# Reads master.csv
allData = pd.read_csv("master.csv")


def get_country(country):
    # Gets data from allData for country specified
    country_data = allData.loc[allData.country == country]

    # Deletes all columns with the names below
    country_data.drop(['country-year', 'HDI for year', 'gdp_for_year ($)', 'gdp_per_capita ($)', 'generation'],
                      axis=1, inplace=True)

    country_data_len = len(country_data)
    first_index = country_data.head(1).index.values.astype(int)[0]  # First year data
    last_index = first_index + country_data_len - 1  # Last year data

    first_year = country_data.loc[first_index]['year']
    last_year = country_data.loc[last_index]['year']

    total_death = []  # y values (suicides/100k pop)
    total_death_index = []  # x values (year)

    for i in range(first_year, last_year + 1):
        try:
            # Adds up suicides/100k pop for all demographics in one year
            total_death.append(country_data.groupby('year').get_group(i)['suicides/100k pop'].sum())
            total_death_index.append(i)  # Appends year value
        except:
            # Needed since data doesn't exist for every year
            pass

    # Return a series with total suicides/100k pop and year
    return pd.Series(total_death, index=total_death_index, name=country)


# Need to find common min year between all countries to not have missing data from any countries
def find_common_min(*country_series):
    common_min_year = 0
    for country in country_series:
        # Finds the first year of collected data for each country
        first_year = country.head(1).index.values.astype(int)[0]

        # If that first year is after the common min year, then it is the new common min year
        if first_year > common_min_year:
            common_min_year = first_year

    return common_min_year


# Also need to find common max year between all the countries to not have missing data
def find_common_max(*country_series):
    common_max_year = 9999  # Should be greater than current year
    for country in country_series:
        first_year = country.tail(1).index.values.astype(int)[0]

        if first_year < common_max_year:
            common_max_year = first_year

    return common_max_year


country = get_country('United States')
common_year_index = []
year_deaths = []

for i in range(find_common_min(country), find_common_max(country)):
    common_year_index.append(i - find_common_min(country))
    # x axis as years since 1985
    try:
        year_deaths.append(np.round(country.loc[i] / 10, decimals=2))
        # y axis tens of deaths
    except:
        pass

z = test.LearnTest(common_year_index, year_deaths, np.array([[0], [0]]), 1)

# contour plot
zoom = 1
fig = plt.figure()
ax = fig.add_subplot(111)
x_space = np.linspace(-25 * zoom, 70 * zoom, 1000)
y_space = np.linspace(-5.0 * zoom, 5.0 * zoom, 1000)
x, y = np.meshgrid(x_space, y_space)
ax.contour(x, y, z.get_cost(x, y), levels=np.arange(-16000, 16000, 300))

# Initial hypothesis
hyp = np.array([[-4.0],  # Initial slope (theta_1)
                [50]])  # Initial x-intercept (theta_0)
z.hypothesis = hyp

z.alpha = 0.001  # Alpha value
theta_0 = []
theta_1 = []

# increase range to make more accurate
for i in range(25000):
    theta_1.append(z.hypothesis[0][0])
    theta_0.append(z.hypothesis[1][0])
    z.adjust_by_gradient()
    i += 1

# Creating the gradient plot
plt.scatter(theta_0, theta_1)
plt.xlabel('x-intercept')
plt.ylabel('slope')
plt.show()

# scatter plot made with final values of theta_1 and theta_0
x = np.linspace(0, 30, 100)
y = theta_1[len(theta_1) - 1] * x + theta_0[len(theta_0) - 1]
plt.plot(x, y)
plt.scatter(common_year_index, year_deaths)
plt.show()
