import csv
import os
from datetime import datetime, timedelta

# Math libraries
import numpy as np
from sklearn.decomposition import fastica, FastICA
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns  # Pretty graphs
import statsmodels.formula.api as smf
from statsmodels.regression.linear_model import RegressionResultsWrapper
from statsmodels.graphics.gofplots import ProbPlot

# Set a date range
start = datetime(
    year=2021,
    month=6,
    day=1
)
end = datetime(
    year=2021,
    month=8,
    day=1
)

usgs_data = {}
discharge_data = []

# Load USGS data
for filename in os.listdir(os.path.join("data", "usgs")):
    if ".csv" in filename:
        with open(os.path.join("data", "usgs", filename), "r", encoding='utf-8-sig') as file:
            file_data = csv.DictReader(file)
            for row in file_data:
                try:
                    site_number = row['site_no']
                    if not site_number == '' and not site_number == ' ':
                        if site_number not in usgs_data.keys():
                            usgs_data[site_number] = []
                        usgs_data[site_number].append(
                            {
                                "datetime": datetime.strptime(row['datetime'], "%m/%d/%y %H:%M"),
                                "Gage height, feet": row['Gage height, feet'],
                            }
                        )
                    else:
                        raise ValueError("End of file, no data")
                except ValueError as err:
                    # Error! Datetime wasn't parsable. Probably because Excel is garbage.
                    # Make sure this is just the last line of each of your files! If not, yikes.
                    print("ERROR:DATETIME FAILED:", row)
                    print("ERROR:", err)

# Load discharge data
with open(os.path.join("data", "DischargeData.csv"), "r", encoding='utf-8-sig') as file:
    file_data = csv.DictReader(file)
    for row in file_data:
        try:
            parsed_datetime = datetime.strptime(f"{row['Sample time']}", "%m/%d/%Y %H:%M")

            minute_rounded = round(parsed_datetime.minute / 15.0) * 15
            hour_mod = 0
            if minute_rounded == 60:
                minute_rounded = 0
                hour_mod = 1

            cleaned_datetime = datetime(
                year=parsed_datetime.year,
                month=parsed_datetime.month,
                day=parsed_datetime.day,
                hour=parsed_datetime.hour,
                minute=minute_rounded,
                second=0
            )

            cleaned_datetime += timedelta(hours=hour_mod)

            discharge_data.append(
                {
                    "datetime": cleaned_datetime,
                    "Discharge": row['Discharge'],
                }
            )
        except ValueError as err:
            # Error! Datetime wasn't parsable. Probably because Excel is garbage.
            # Make sure this is just the last line of each of your files! If not, yikes.
            print("ERROR:DATETIME FAILED:", row)
            print("ERROR:", err)

# Sort USGS data
for site in usgs_data.keys():
    usgs_data[site] = sorted(usgs_data[site], key=lambda d: d['datetime'])
# Sort Discharge data
discharge_data = sorted(discharge_data, key=lambda d: d['datetime'])

# Only include dates between start and end
for site in usgs_data.keys():
    usgs_data[site] = [x for x in usgs_data[site] if start < x['datetime'] < end]
discharge_data = [x for x in discharge_data if start < x['datetime'] < end]

# Put discharge data in USGS data
merged_data = {}
for site in usgs_data.keys():
    merged_data[site] = []
    for usgs_obj in usgs_data[site]:
        for discharge_obj in discharge_data:
            # print(usgs_obj['datetime'] == discharge_obj['datetime'], usgs_obj['datetime'], discharge_obj['datetime'])
            if usgs_obj['datetime'] == discharge_obj['datetime']:
                merged_data[site].append((
                    usgs_obj['datetime'].timestamp(),
                    usgs_obj['datetime'],
                    usgs_obj['Gage height, feet'],
                    discharge_obj['Discharge'],
                ))
                break
    print(site, len(merged_data[site]))

# FastICA arrays
timestamp_arr = [x[0] for x in merged_data['2172035']]
datetime_arr = [x[1] for x in merged_data['2172035']]
gage_arr = [x[2] for x in merged_data['2172035']]
discharge_arr = [x[3] for x in merged_data['2172035']]

# Relevant series with datetime
gage_series = pd.Series(data=gage_arr, index=datetime_arr)
discharge_series = pd.Series(data=discharge_arr, index=datetime_arr)

# DataFrame with the time series
x = pd.DataFrame({'discharge': discharge_series, 'gage': gage_series})

# FastICA model that we will fit our data into
ica = FastICA(
    n_components=2,
    algorithm="parallel",
    fun="logcosh",
    fun_args={
        "alpha": 1.0
    },
    max_iter=200,
    tol=0.00001,
)
# Fit that data
S_ = ica.fit_transform(x)

# Fit result of ICA to the series
S_series1 = pd.Series(data=S_[:, 0], index=datetime_arr)
S_series2 = pd.Series(data=S_[:, 1], index=datetime_arr)

# Set up the plotting library
plt.style.use('seaborn') # pretty matplotlib plots
plt.rc('font', size=14)
plt.rc('figure', titlesize=18)
plt.rc('axes', labelsize=15)
plt.rc('axes', titlesize=18)

"""Do linear regression"""
# Linear Regression math
model_f = 'discharge ~ s1 + s2'

# Create New DataFrame with:
# - Gage
# - Discharge
# - Signal 1 from FastICA
# - Signal 2 from FastICA
fitted_df = x
fitted_df['s1'] = S_[:, 0]
fitted_df['s2'] = S_[:, 1]
fitted_df['discharge'] = pd.to_numeric(fitted_df['discharge'])
fitted_df['gage'] = pd.to_numeric(fitted_df['gage'])
fitted_df['datetime'] = pd.to_datetime(datetime_arr)

# Drop NA
fitted_df.dropna(inplace=True)

# IDK but this helped
fitted_df.reset_index(drop=True, inplace=True)

# This is the actual LinearRegression
model = smf.ols(formula=model_f, data=fitted_df)
model_fit: RegressionResultsWrapper = model.fit()

# Coefficients (Intersect [0], signal coefficient [1], signal coefficient [2])
coefs = model_fit.params

# Create the results
baseflow_tide = x['s1'] * coefs[1] + coefs[0]
baseflow_tide_runoff = (
        (
                coefs[1] * fitted_df['s1']
        ).add(coefs[0])
).add(
    coefs[2] * fitted_df['s2']
)

# Plot the data
plt.plot(fitted_df[['discharge']])
plt.plot(baseflow_tide)
plt.plot(baseflow_tide_runoff)
plt.show()

# Export the data
datetime_final = fitted_df['datetime'].values.tolist()
discharge_final = fitted_df['discharge'].values.tolist()
baseflow_tide_final = baseflow_tide.values.tolist()
baseflow_tide_runoff_final = baseflow_tide_runoff.values.tolist()
final_data = []
for i in range(len(datetime_final)):
    final_data.append({
        'datetime': datetime_final[i],
        'discharge': discharge_final[i],
        'baseflow_tide': baseflow_tide_final[i],
        'baseflow_tide_runoff': baseflow_tide_runoff_final[i],
    })

print(final_data[0])

with open('exported_data.csv', 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=final_data[0].keys())
    writer.writeheader()
    writer.writerows(final_data)

