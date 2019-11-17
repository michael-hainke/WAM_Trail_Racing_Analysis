# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 19:56:10 2018

@author: T887343
"""

import numpy as np # matrix functions
import pandas as pd # data manipulation
import requests # http requests
from bs4 import BeautifulSoup # html parsing
import matplotlib.pyplot as plt # plotting
import seaborn as sns # plotting
import os # change directory
import re # replace stings

# Read WAM Data from Results Webpage
################################

def read_results(url):
    res = requests.get(url) 
    soup = BeautifulSoup(res.content,'lxml') 
    table = soup.find_all('table')[0]  
    df = pd.read_html(str(table)) 
    return pd.DataFrame(df[0]) 

url = "http://www.racesplitter.com/races/204FD653A" #2016 data
df_2016 = read_results(url)
df_2016['Year'] = 2016

url = "http://www.racesplitter.com/races/EC4DC29E6" #2017 data
df_2017 = read_results(url)
df_2017['Year'] = 2017

df = pd.concat([df_2016, df_2017])

# Process Data to add Total Time in Minutes
###########################################
def convert_to_minutes(row):
    return sum(i*j for i, j in zip(map(float, row['Time'].split(':')), [60, 1, 1/60]))
df['Minutes'] = df.apply(convert_to_minutes, axis=1)

# write result data to csv
os.chdir(r"C:\Users\T887343\Desktop\WIP\Courses\Python\Test Code")
df.to_csv('18-07-22 WAM Results.csv')

# Add in 2018 Race Data
os.chdir(r"C:\Users\T887343\Desktop\WIP\Git Repositories\WAM_Trail_Racing_Analysis")
df = pd.read_csv("18-07-22 WAM Results.csv")

url = "https://spruceregistrations.com/coast-mountain-trail-series/races/79/results?categorization=27" #2018 data
df_2018 = read_results(url)
df_2018.columns = ['Pos','Bib','Name','Time']
df_2018 = df_2018.iloc[2:214,]
df_2018.Time = df_2018.Time.str.replace('\s\S+', '')
df_2018['Minutes'] = df_2018.apply(convert_to_minutes, axis=1)
df_2018['Year'] = 2018

df = pd.concat([df, df_2018])

# Load 2018 Race data - my age group only
url = "https://spruceregistrations.com/coast-mountain-trail-series/races/79/results?categorization=17" #2018 data
df_2018_splits = read_results(url)
df_2018_splits.columns = ['Pos','Bib','Name','Time']

start = [i for i, j in enumerate(df_2018_splits.Pos) if j == '25k Men 40 to 49'][0] + 2
end = [i for i, j in enumerate(df_2018_splits.Pos) if j == '25k Women 50 to 59'][0]

df_2018_splits = df_2018_splits.iloc[start:end,]

df_2018_splits['me'] = 'Not me'
df_2018_splits.me.iloc[26] = 'Me'

df_2018_splits.Time = df_2018_splits.Time.str.replace('\s\S+', '')
df_2018_splits['Minutes'] = df_2018_splits.apply(convert_to_minutes, axis=1)

# Read Scotia Half data from Results Webpage
############################################

#os.chdir(r"C:\Users\T887343\Desktop\WIP\Courses\Python\Test Code")
#scotia_2015 = pd.read_csv("ScotiaHalf2015.csv")
#
## select only relevant columns and remove NAs
#scotia_2015 = scotia_2015.loc[:,['Chip_Time','Gender','Age_Group']].dropna()
#
#def convert_to_minutes(row):
#    return sum(i*j for i, j in zip(map(float, row['Chip_Time'].split(':')), [60, 1, 1/60]))
#
#scotia_2015['Minutes'] = scotia_2015.apply(convert_to_minutes, axis=1)
#
## remove outliers > 3hrs 30 min (210 min)
#scotia_2015 = scotia_2015[scotia_2015['Minutes'] < 210]
#
#sns.violinplot(x="Gender", y="Minutes", data=scotia_2015, inner=None)
#
#sns.violinplot(x="Age_Group", y="Minutes", data=scotia_2015[scotia_2015['Gender'] == 'Male'], inner=None )

# Plot Data
###########

sns.violinplot(x="Gender", y="Minutes", data=df, inner=None)
sns.swarmplot(x="Gender", y="Minutes", data=df, color="w", alpha=.5)

plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
sns.swarmplot(x="Gender", y="Minutes", data=df)
plt.title('WAM Split by Gender', fontsize=18, fontweight="bold")
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.xlabel("Gender", fontsize=18)
plt.ylabel("Minutes", fontsize=18)
plt.savefig("WAM SwarmPlot.png")

plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
sns.boxplot(x="Year", y="Minutes", data=df)
plt.title('WAM Results by Year', fontsize=18, fontweight="bold")
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.xlabel("Year", fontsize=18)
plt.ylabel("Minutes", fontsize=18)
plt.savefig("WAM BoxPlot.png")

plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
sns.violinplot(x="Year", y="Minutes", data=df, inner='quartile')
plt.title('WAM Results by Year', fontsize=18, fontweight="bold")
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.xlabel("Year", fontsize=18)
plt.ylabel("Minutes", fontsize=18)
plt.savefig("WAM ViolinPlot.png")

plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
#sns.swarmplot(x="Year", y="Minutes", data=df, hue='Gender')
#sns.swarmplot(x="Year", y="Minutes", data=df) # for 2018 data with no gender
sns.violinplot(x="Year", y="Minutes", data=df, inner='quartile')
plt.title('WAM Results by Year', fontsize=18, fontweight="bold")
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.xlabel("Year", fontsize=18)
plt.ylabel("Minutes", fontsize=18)
#plt.savefig("WAM SwarmPlot.png")
plt.savefig("WAM ViolinPlot 2018.png")

# subset only men's results
men = df.loc[df['Gender'] == 'M']
men['Age Group'] = men['Age Group'].cat.remove_unused_categories()
# plot violin and swarm plots by age group
plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
sns.violinplot(x="Age Group", y="Minutes", data=men, color='lightblue', inner='quartile')
sns.swarmplot(x="Age Group", y="Minutes", data=men, color='darkblue')
plt.title('Mens WAM Results by Age Group', fontsize=18, fontweight="bold")
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.xlabel("Age Group", fontsize=18)
plt.ylabel("Minutes", fontsize=18)
plt.savefig("WAM Mens SwarmPlot.png")

# subset only women's results
women = df.loc[df['Gender'] == 'F']
women['Age Group'] = women['Age Group'].cat.remove_unused_categories()
# plot violin and swarm plots by age group
plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
sns.violinplot(x="Age Group", y="Minutes", data=women, color='wheat', inner='quartile')
sns.swarmplot(x="Age Group", y="Minutes", data=women, color='darkorange')
plt.title('Womens WAM Results by Age Group', fontsize=18, fontweight="bold")
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.xlabel("Age Group", fontsize=18)
plt.ylabel("Minutes", fontsize=18)
plt.savefig("WAM Womens SwarmPlot.png")

# Calculate Summary Statistics

# subset my age group results
group_times = df['Minutes'].where(df['Age Group'] == 'M 40-49').dropna()
# 25, 50 and 75 percentiles for total time and calculated per km pace
np.round(np.percentile(group_times, [25, 50, 75]), 1)
np.round(np.percentile(group_times, [25, 50, 75]) / 25, 1)

# plot 2018 results for my age group
plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
sns.swarmplot(y="Minutes", data=df_2018_splits[(df_2018_splits.Name != 'Michael Hainke')], color='lightblue')
sns.swarmplot(y="Minutes", data=df_2018_splits[(df_2018_splits.Name == 'Michael Hainke')], color='purple')
plt.title('2018 WAM Results M 40-49', fontsize=18, fontweight="bold")
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.ylabel("Minutes", fontsize=18)
plt.savefig("WAM SwarmPlot 2018 age group.png")

np.round(np.percentile(df_2018_splits.Minutes, [25, 50, 75]), 1)

# RANDOM SCRIPT
###############

df['Gender'].value_counts()

# Histogram for results
plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
plt.hist(df['Minutes'], facecolor='blue', alpha=0.5)
plt.title('2016 & 2017 Whistler Alpine Meadows Times', fontsize=18, fontweight="bold")
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.xlabel('Time (min)', fontsize=18)
plt.ylabel('Frequency', fontsize=18)

plt.savefig("WAM Histogram.png")

plt.show()

# List of genders to plot
genders = ['M','F']

# Iterate through the genders
for gender in genders:
    # Subset to the gender
    subset = df[df['Gender'] == gender]
    
    # Draw the density plot
    sns.distplot(subset['Minutes'], hist = True, kde = True, bins=15,
                 kde_kws = {'shade': True, 'linewidth': 3},
                 label = gender)
    
# Plot formatting
plt.legend(prop={'size': 16})
plt.title('2017 Whistler Alpine Meadows Times')
plt.xlabel('Time (min)')
plt.ylabel('Frequency')
           
plt.show()

# List of age groups to plot
age_groups = ['M 20-29','M 30-39', 'M 40-49', 'M 50-59', 'M 60+']

# Iterate through the genders
for age_group in age_groups:
    # Subset to the gender
    subset = df[df['Age Group'] == age_group]
    
    # Draw the density plot
    sns.distplot(subset['Minutes'], hist = True, kde = False,
                 kde_kws = {'shade': True, 'linewidth': 3},
                 label = age_group)
    
# Plot formatting
plt.legend(prop={'size': 16})
plt.title('2017 Whistler Alpine Meadows Times')
plt.xlabel('Time (min)')
plt.ylabel('Density')
           
plt.show()

# Access Strava data using API
###############################

access_token = "access_token=1e976a06f9636caed22ededb5315e2832a31016b" # enter your access code here


# get specific activity
url = "https://www.strava.com/api/v3/activities/1712099212?access_token="


## get list of all activities

# initialize variables
url = "https://www.strava.com/api/v3/activities"
page = 1
col_names = ['id','type']
activities = pd.DataFrame(columns=col_names)

while True:
    
    # get page of activities from Strava
    r = requests.get(url + '?' + access_token + '&per_page=50' + '&page=' + str(page))
    r = r.json()

    # if no results then exit loop
    if (not r):
        break
    
    # otherwise add new data to dataframe
    for x in range(len(r)):
        activities.loc[x + (page-1)*50,'id'] = r[x]['id']
        activities.loc[x + (page-1)*50,'type'] = r[x]['type']

    # increment page
    page += 1

# barchart of activity types
activities['type'].value_counts().plot('bar')
plt.title('Activity Breakdown', fontsize=18, fontweight="bold")
plt.xticks(fontsize=14)
plt.yticks(fontsize=16)
plt.ylabel('Frequency', fontsize=18)

 
## get splits from all runs
# filter to only runs
runs = activities[activities.type == 'Run']

# initialize dataframe for split data
col_names = ['average_speed','distance','elapsed_time','elevation_difference','moving_time','pace_zone',
             'split','id','date','description']
splits = pd.DataFrame(columns=col_names)

# loop through each activity id and retrieve data
for run_id in runs['id']:
    
    # Load activity data
    print(run_id)
    r = requests.get(url + '/' + str(run_id) + '?' + access_token)
    r = r.json()

    # Extract Activity Splits
    activity_splits = pd.DataFrame(r['splits_metric']) 
    activity_splits['id'] = run_id
    activity_splits['date'] = r['start_date']
    activity_splits['description'] = r['description']
    
    # Add to total list of splits
    splits = pd.concat([splits, activity_splits])


# write split data to csv
os.chdir(r"C:\Users\T887343\Desktop")
splits.to_csv('18-08-25 New Activity Splits.csv')

# read in data from csv
splits = pd.read_csv('18-08-25 New Activity Splits.csv')

# Histogram for split distances
plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
plt.hist(splits['distance'], facecolor='blue', alpha=0.5)
plt.title('Split Distances', fontsize=18, fontweight="bold")
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.xlabel('Distance (m)', fontsize=18)
plt.ylabel('Frequency', fontsize=18)

# Filter to only those within +/-50m of 1000m
splits = splits[(splits.distance > 950) & (splits.distance < 1050)]

# Scatter plot of elevation vs. pace
plt.plot( 'elevation_difference', 'moving_time', data=splits, linestyle='', marker='o', markersize=3, alpha=0.1, color="blue")
plt.title('Running Pace vs. Elevation Change', fontsize=18, fontweight="bold")
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.xlabel('Elevation Change (m)', fontsize=18)
plt.ylabel('1km Pace (sec)', fontsize=18)

# Calculation of difference between elapsed and moving time and plot
splits['time_diff'] = splits['elapsed_time'] - splits['moving_time']

plt.plot( 'elevation_difference', 'time_diff', data=splits, linestyle='', marker='o', markersize=3, alpha=0.1, color="blue")
plt.title('Time Difference vs. Elevation Change', fontsize=18, fontweight="bold")
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.xlabel('Elevation Change (m)', fontsize=14)
plt.ylabel('Elapsed Time - Moving Time (sec)', fontsize=14)


# Plot scatter plot with trendline
sns.regplot(x = 'elevation_difference', y = 'elapsed_time', data = splits ,order = 2)
plt.title('Running Pace vs. Elevation Change', fontsize=18, fontweight="bold")
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.xlabel('Elevation Change (m)', fontsize=18)
plt.ylabel('1km Pace (sec)', fontsize=18)

# Plot scatter with plot function and trendline with regplot
plt.plot( 'elevation_difference', 'elapsed_time', data=splits, linestyle='', marker='o', markersize=5, alpha=0.1, color="blue")
sns.regplot(x = 'elevation_difference', y = 'elapsed_time', scatter=None, data = splits ,order = 2)
plt.title('Running Pace vs. Elevation Change', fontsize=18, fontweight="bold")
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.xlabel('Elevation Change (m)', fontsize=18)
plt.ylabel('1km Pace (sec)', fontsize=18)

# Get the equation for the fitted line
coeff = np.polyfit(splits['elevation_difference'], splits['elapsed_time'], 2)
 
# Calculate estimated time from model
WAM = pd.read_csv('WAM_25k_course.csv')

WAM['estimated_time'] = coeff[0]*WAM['elevation']**2 + coeff[1]*WAM['elevation'] + coeff[2]
WAM['estimated_time'].sum() / 60

# filter to only trail, and road runs
splits_trail = splits[splits['description'] == 'Trail']
coeff_trail = np.polyfit(splits_trail['elevation_difference'], splits_trail['elapsed_time'], 2)
WAM['estimated_time_trail'] = coeff_trail[0]*WAM['elevation']**2 + coeff_trail[1]*WAM['elevation'] + coeff_trail[2]
WAM['estimated_time_trail'].sum() / 60

