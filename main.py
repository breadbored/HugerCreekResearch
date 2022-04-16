import csv
import os
from datetime import datetime, timedelta

# Math libraries
import numpy as np
from sklearn.decomposition import fastica, FastICA
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd

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

# print(len(usgs_data))
# print(len(discharge_data))
# print(len(merged_data))

# NEW FastICA RESHAPING
timestamp_arr = [x[0] for x in merged_data['2172035']]
datetime_arr = [x[1] for x in merged_data['2172035']]
gage_arr = [x[2] for x in merged_data['2172035']]
discharge_arr = [x[3] for x in merged_data['2172035']]

time = np.linspace(0, len(timestamp_arr) - 1, len(timestamp_arr))

# md_array = [
#     timestamp_arr,
#     gage_arr,
#     discharge_arr
# ]
x = pd.DataFrame({'discharge': discharge_arr, 'gage': gage_arr})

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

S_ = ica.fit_transform(x)
A_ = ica.mixing_
print("x: ", len(x),  x)
print("S_:", len(S_), S_)
print("A_:", len(A_), A_)

plt.figure(figsize=(9, 6))

model = LinearRegression()
model.fit(S_, x[['discharge']])
print("S model coef:", model.coef_)

# x['datetime'] = datetime_arr
# x['timestamp'] = timestamp_arr

plt.hist(x[['discharge']].values)
plt.hist(model.predict(x))
plt.show()

# x['datetime'] = datetime_arr
# x['timestamp'] = timestamp_arr
# plt.scatter(x['datetime'], x['discharge'], color='red')
# plt.plot(x['datetime'], model.predict(x['datetime']), color='blue')
# plt.title('Salary vs Experience')
# plt.xlabel('Years of Experience')
# plt.ylabel('Salary')
# plt.show()

# OLD WAY
# md_array = [
#     datetime_arr,
#     gage_arr,
#     discharge_arr
# ]
#
# K, W, S, X_mean = fastica(
#     X=md_array,
#     n_components=3,
#     algorithm="parallel",
#     fun="logcosh",
#     fun_args={
#         "alpha": 1.0
#     },
#     max_iter=200,
#     tol=0.00001,
#     return_X_mean=True
# )
#
# print("K:", K)
# print("W:", W)
# print("S:", S)
# print("X_mean:", X_mean)
#
# print("Shape1:", S.shape)
# print("Shape2:", len(md_array[-1]))
#
# model = LinearRegression()
# model.fit(S, md_array[-1])
#
# plt.scatter(S[0] + S[1], md_array[-1], color='red')
# plt.plot(S[0] + S[1], model.predict(S[0] + S[1]), color='blue')
#
# plt.title('TITLE')
# plt.xlabel('X LABEL')
# plt.ylabel('Y LABEL')
# plt.show()
# plt.plot(*result)
