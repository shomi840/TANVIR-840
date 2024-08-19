# -*- coding: utf-8 -*-
"""Copy of Week_05_Visualization_Techniques.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1w_brAzVNqJ9MPQhRRlbyGQAZMMo-nPXq

# Week 05: Visualization Techniques
**Course:** WMASDS04 - Introduction to Data Science with Python
<br>**Instructor:** Farhana Afrin, Department of Statistics, JU

**Outlines:**
- Pie Chart
- Bar Chart
- Scatter Plot
- Heatmap
- Line Plot
- Histogram
- Box Plot
- Violin Plot
- Time Series Plot

#### Import required libraries and packages
"""

# for data analysis
import numpy as np
import pandas as pd

# for data visualization
import matplotlib.pyplot as plt
import seaborn as sns

"""#### Read a csv file as pandas DataFrame"""

df = pd.read_csv('bmw.csv')



import pandas as pd
import numpy as np

pd.read_csv('bmw.csv')

# print(type(df))
# display(df.head())
# display(df.tail())
# df.info()
df.dtypes
# df.sample(5,random_state = 4)
# df.columns
# df.index
df.shape

print('Number of Rows = ', df.shape[0])
print('Number of Columns = ', df.shape[1])

"""#### Checking for missing values"""

# df.info()
df.isnull().sum()
df.notnull().sum()

"""#### Count the number of unique values in columns"""

# df['model'].unique()
# df['transmission'].value_counts()
df['fuelType'].value_counts()

# print(df['model'].unique())
print(type(df['model'].unique()))
len(df['model'].unique())

"""## Pie chart"""

df['transmission'].value_counts().plot.pie(autopct = '%.2f%%')

df['transmission'].value_counts().plot(kind = 'pie',
                                      autopct = '%.1f%%')
plt.ylabel('')
plt.show()

type(df['transmission'].value_counts())



"""## Bar Plot

- How to set a title, xlabel and ylabel of a plot?
- How to change the range of x and y axis?

#### Find the percentage of unique values present in the 'fuelType' column
"""

print(df["fuelType"].value_counts())

print(type(df["fuelType"].value_counts()))

df_fuelType = pd.DataFrame(df["fuelType"].value_counts())

display(df_fuelType.head())
print(df_fuelType.index)
print(df_fuelType.columns)

df_fuelType = pd.DataFrame(df['fuelType'].value_counts())
df_fuelType = df_fuelType.reset_index()
df_fuelType = df_fuelType.rename(columns = {'index':'fuelType',
                                           'fuelType':'no_of_cars'})
df_fuelType['% of cars'] = (df_fuelType['no_of_cars']/df.shape[0])*100

# df_fuelType['no_of_cars'].sum()
# df.shape[0]

df_fuelType

"""#### Barplot for the 'fuelType' column"""

sns.barplot(x="fuelType",
            y="% of cars",
            data=df_fuelType,
            color="red",
            alpha=0.8)

plt.xlabel("Types of fuel")
plt.ylabel("% of cars")
plt.title("Percentage of cars present in each fuelType")

plt.yticks(np.arange(0,101,10))

plt.show()

"""**barplot for model column**"""

df_model = pd.DataFrame(df['model'].value_counts())
df_model = df_model.reset_index()
df_model = df_model.rename(columns = {'index': 'Model','model': 'No_of_car'})

# df_model.columns
# df_model.index
df_model

plt.bar(df_model['Model'], df_model['No_of_car'])
plt.barh(df_model['Model'], df_model['No_of_car'])

plt.figure(figsize = (30,8))
sns.barplot(x = 'Model', y= 'No_of_car',color = 'blue', data = df_model, alpha = 0.5)

plt.figure(figsize = (20,8))
df_model = df_model.sort_values(by = ['No_of_car'], ascending = True)
plt.barh(df_model['Model'], df_model['No_of_car'], height = 0.8, left = 0)
plt.xticks(np.arange(0,2501, 500))
plt.tight_layout()
plt.show()

plt.rcParams.update({'font.size': 16})

plt.figure(figsize = (20,8))
df_model = df_model.sort_values(by = ['No_of_car'], ascending = True)
plt.barh(df_model['Model'], df_model['No_of_car'], height = 0.8, left = 0)

plt.tight_layout()
plt.show()

plt.barh(df_model['Model'], df_model['No_of_car'])

"""## Scatter Plot

#### Find the relation between the numerical variables

- Scatterplot of mileage vs price
- Scatterplot of mpg vs price
- Scatterplot of engineSize vs price
"""

df.dtypes

df.describe()

plt.figure(figsize=(20,16))

plt.subplot(3, 1, 1)
sns.scatterplot(x="mileage", y="price", data=df)

plt.subplot(3, 1, 2)
sns.scatterplot(x="mpg", y="price", data=df)

plt.subplot(3, 1, 3)
sns.scatterplot(x="engineSize", y="price", data=df)

plt.tight_layout()
plt.show()

"""#### using the hue parameter"""

plt.figure(figsize=(20,16))

plt.subplot(2, 2, 1)
sns.scatterplot(x="mileage", y="price", data=df, hue="fuelType")

plt.subplot(2, 2, 2)
sns.scatterplot(x="mpg", y="price", data=df, hue="fuelType")

plt.subplot(2, 2, 3)
sns.scatterplot(x="engineSize", y="price", data=df, hue="fuelType")

plt.tight_layout()
plt.show()

# Addding trend line
plt.figure(figsize=(20,16))

plt.subplot(2, 2, 1)
sns.regplot(x="mileage", y="price", data=df)

plt.subplot(2, 2, 2)
sns.regplot(x="mileage", y="price", data=df, line_kws={"color":"red"})

plt.subplot(2, 2, 3)
sns.regplot(x="mileage", y="price", data=df, scatter_kws={"color":"orange", "edgecolor":"white"})

plt.tight_layout()
plt.show()



"""### Pairplot"""

# # an example from the documentation
# penguins = sns.load_dataset("penguins")
# display(penguins.head())
# print(penguins.shape)
# # sns.pairplot(penguins)
# sns.pairplot(penguins, hue="species")
# plt.show()

# sns.pairplot(df, corner=True)
sns.pairplot(df, corner=True, hue="transmission")



"""## Heatmap"""

df.info()

"""#### Assumptions:

- **Car price increases when engineSize increases**
- **Car price increases when mpg increases**
- **Car price decreases when the mileage increases**
- **When the engineSize increases, the mpg decreases**
- **Car price increases with the latest year cars**
"""

df.corr()

sns.heatmap(df.corr())

sns.heatmap(df.corr(), square=True, vmax=1.0, vmin=-1.0, cmap="RdYlGn", annot=True)

# Creating mask
correlation_matrix = df.corr()
display(correlation_matrix)

mask = np.zeros_like(correlation_matrix)
print(mask)

mask[np.triu_indices_from(mask)] = True
print(mask)

plt.figure(figsize=(10,10))
sns.heatmap(correlation_matrix,
            square=True,
            vmax=1.0, vmin=-1.0,
            cmap="RdYlGn",
            annot=True,
            mask=mask)

plt.title("Heatmap of the Pearson Correlation Coefficient")
plt.show()

"""#### observations:

- **Car price increases when engineSize increases:** True
- **Car price increases when mpg increases**: False
- **Car price decreases when the mileage increases**: True
- **When the engineSize increases, the mpg decreases**: True
- **Car price increases with the latest year cars** : True
"""





"""### Line Plot
- How to create a user defined function?
- How to draw the equation of a straight line?
- How to draw multiple line charts in the same figure?
- How to change the color, linestyle and marker of a figure?
- How to modify the legend of a figure?
- How to create Subplot?

"""



m = -1
c = 0
x = np.arange(-10,11,2)
y = m*x+c
plt.plot(x,y)

# Subplot with same dataset formated differently

plt.figure(figsize=(16,8))
plt.subplot(2,2,1)
plt.plot(x,y)
plt.title('this is the default line')

plt.subplot(2,2,2)
plt.plot(x,y, color = 'Red')
plt.title(' This is customized red line')

plt.subplot(2,2,3)
plt.plot(x,y, color = 'Red', marker = 'o')
plt.title('This figure adds a marker')

plt.subplot(2,2,4)
plt.plot(x,y, color = 'Green', marker = 'o')
plt.hlines(0,-10,10, color = 'Purple', alpha = .8)
plt.vlines(0,-10,10)
plt.xlabel('X variable')
plt.ylabel('Y variable')
plt.title('This looks better!')
plt.legend(['line', 'x line', 'y line'], loc = 'upper right')

plt.suptitle('This are the examples of line diagram')
plt.show()

# Formating plot: linestyle, color, marker, title, label, gridlines, axes ticks, legend
m = -1
c = 0
x = np.arange(-10,11,2)
y = m*x+c

plt.figure(figsize = (6,4))
plt.plot(x,y, color = 'Red', marker = 'o')

plt.hlines(0, -10, 10, color="green", linestyles="-")
plt.vlines(0, -15, 15, color="blue", linestyles="-")

plt.xticks(np.arange(-10,11,5))
plt.yticks(np.arange(-15,16,10))
plt.minorticks_on()

plt.grid(True, which='both')
plt.legend(['line', 'x line', 'y line'], loc = 'lower left')

plt.xlabel("This is X axis")
plt.ylabel("This is Y axis")

plt.title('This is a line diagram')
plt.tight_layout()
plt.show()

# Creating multiple straight lines using function
def line(x, m, c):
    df = pd.DataFrame() #creating blank dataframe
    df['x'] = x         # add a column, x
    df['y'] = m*x+c
    return df

l1 = line(np.arange(-5,6,1), 1, 1)
l2 = line(np.arange(-5,6,1), -1, 1)
l3 = line(np.arange(-5,6,1), 1, -1)
l4 = line(np.arange(-5,6,1), -1, -1)
# l3

# Create a lineplot function
def lineplot(x,y, title, color):
    plt.plot(x,y,color = color, marker= '.')
    plt.hlines(0,-5,6, color = 'Black')
    plt.vlines(0, -5, 6, color = 'Black')
    plt.xticks(np.arange(-6, 7, 2))
    plt.yticks(np.arange(-6, 7, 2))
    plt.minorticks_on()
    plt.grid(True, which = 'both')
    plt.xlabel('x values')
    plt.ylabel('y values')
    plt.title(title)

# Creating subplots
plt.figure(figsize = (10,10))
plt.subplot(2,2,1)
lineplot(l1['x'],l1['y'],'y = mx+c; m=1,c=1','blue')

plt.subplot(2,2,2)
lineplot(l2['x'],l2['y'],'y = mx+c; m=-1,c=1','green')

plt.subplot(2,2,3)
lineplot(l3['x'],l3['y'],'y = mx+c; m=1,c=-1','black')

plt.subplot(2,2,4)
lineplot(l4['x'],l4['y'],'y = mx+c; m=-1,c=-1','purple')

plt.suptitle('Line plots for different value of m and c')
plt.tight_layout()
plt.show()

# Multiple lineplots in the same figure

plt.figure(figsize = (10,7))

plt.plot(l1['x'],l1['y'], label='y = mx+c; m=1,c=1',color='blue')
plt.plot(l2['x'],l2['y'],label='y = mx+c; m=-1,c=1',color='green')
plt.plot(l3['x'],l3['y'],label = 'y = mx+c; m=1,c=-1',color='red')
plt.plot(l4['x'],l4['y'],label='y = mx+c; m=-1,c=-1',color='purple')
plt.vlines(0, -5, 6, linestyles=':', colors="black")
plt.hlines(0, -5, 6, linestyles=':', colors="black")
plt.title("y=mx+c")
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc="center right")
plt.grid()
plt.show()





"""## Histogram, Distribution, ECDF

Dataset: cellular usage dataset that consists of records of actual cell phone that include specific features such as

1. **Account_Length**: the number of days the customer has the subscription with the telecom company

2. **Vmail_Message**: the total number of voicemails the customer has sent

3. **Total_mins**: the total number of minutes the customer has talked over the phone

4. **CustServ_Calls**: the number of customer service calls the customer made

5. **Churn**: yes and no - indicating whether or not the customer has churned

6. **Intl_Plan**: yes and no - indicating whether or not the customer has international plan or not

7. **Vmail_Plan**: yes and no - indicating whether or not the customer has voicemail plan or not

8. **Total_calls**: the total number of calls the customer has made

9. **Total_charges**: the total amount of bill in $ the customer has paid
"""

df = pd.read_csv("telecom_data.csv")

display(df.head())
# print(df.info())
# print(df.shape)
# display(df.describe())
# display(df.describe(include=['O']))

"""## Histogram"""

sns.histplot(x="Total_charges", data=df, binwidth = 10, stat = 'probability')



plt.figure(figsize=(20,12))

plt.subplot(2,3,1)
sns.histplot(x="Total_charges", data=df)
plt.title("row=1, col=1, position=1")

plt.subplot(2,3,2)
sns.histplot(x="Total_charges", data=df, binwidth=50)
plt.title("row=1, col=2, position=2")

plt.subplot(2,3,3)
sns.histplot(x="Total_charges", data=df, binwidth=50)
plt.xticks(np.arange(df["Total_charges"].min(), df["Total_charges"].max()+51, 50), rotation=90)
plt.title("row=1, col=3, position=3")

plt.subplot(2,3,4)
sns.histplot(x="Total_charges", data=df, binwidth=25, color="orange")
plt.title("row=2, col=1, position=4")

plt.subplot(2,3,5)
sns.histplot(x="Total_charges", data=df, binwidth=25, color="orange", stat="probability")
plt.title("row=2, col=2, position=5")

plt.subplot(2,3,6)
sns.histplot(x="Total_charges", data=df, binwidth=25, color="orange", fill=False)
plt.title("row=2, col=3, position=6")

plt.tight_layout()
plt.show()



plt.figure(figsize=(20,16))

plt.subplot(2,2,1)
sns.histplot(x="Total_charges", data=df, binwidth=1)
plt.title("row=1, col=1, position=1")

plt.subplot(2,2,2)
sns.histplot(x="Total_charges", data=df, binwidth=10, cumulative=True)
plt.title("row=1, col=2, position=2")

plt.subplot(2,2,3)
sns.histplot(x="Total_charges", data=df, binwidth=50, color="green")
plt.xticks(np.arange(df["Total_charges"].min(), df["Total_charges"].max()+51, 50), rotation=90)
plt.grid()
plt.yticks(np.arange(0,4000,250))
plt.title("row=2, col=1, position=3")

plt.subplot(2,2,4)
sns.histplot(x="Total_charges", data=df, binwidth=50, cumulative=True, color="orange")
plt.yticks(np.arange(0,4000,250))
plt.xticks(np.arange(df["Total_charges"].min(), df["Total_charges"].max()+51, 50), rotation=90)
plt.grid()
plt.title("row=2, col=2, position=4")

plt.tight_layout()
plt.show()



plt.figure(figsize=(20,16))

plt.subplot(2,2,1)
sns.histplot(x="Total_charges", data=df, binwidth=1, kde=True)
plt.title("row=1, col=1, position=1")

plt.subplot(2,2,2)
sns.histplot(x="Total_charges", data=df, binwidth=1, kde=True, hue="Churn", stat = "probability")
plt.title("row=1, col=2, position=2")

plt.tight_layout()
plt.show()



df['Total_charges'].sort_values()[1:2]



"""## Box plot"""



plt.figure(figsize=(8,5))
sns.boxplot(y = "Total_charges", data=df, showfliers=False)

"""1. min,
2. first quartile (25th percentile),
3. median (50th percentile)
4. third quartile (75th percentile)
5. max
- range = max-min
- IQR (inter-quartile range) = third quartile - first quartile
"""

# separate numerical columns
num_cols = []

for col in df.columns:
    if df[col].dtypes != "O":
        num_cols.append(col)
#         print(num_cols)

print(num_cols)

plt.figure(figsize=(20,8)) # width, height

for index in range(len(num_cols)):

#     print("index = ", index, "position = ", index+1, "column name = ", num_cols[index])

    plt.subplot(2,3,index+1)
    sns.boxplot(x=num_cols[index], data=df, showfliers=False)

plt.suptitle("Boxplot of the numeric columns in the telecom dataset")
plt.tight_layout()
plt.show()



# Adding categorical features

plt.figure(figsize=(20,8)) # width, height

for index in range(len(num_cols)):
    plt.subplot(2,3,index+1)
    sns.boxplot(x=num_cols[index], data=df, showfliers=False, y=df["Churn"])

plt.suptitle("Boxplot of the numeric columns in the telecom dataset")
plt.tight_layout()
plt.show()



plt.figure(figsize=(25,12)) # width, height

for index in range(len(num_cols)):
    plt.subplot(2,3,index+1)
    sns.boxplot(y=df[num_cols[index]], x=df["Intl_Plan"], hue=df["Churn"], showfliers=False)

plt.suptitle("Boxplot of the numeric columns in the telecom dataset")
plt.tight_layout()
plt.show()



"""## Violin Plot"""

plt.figure(figsize=(20,8)) # width, height

for index in range(len(num_cols)):
    plt.subplot(2,3,index+1)
    sns.violinplot(x=num_cols[index], data=df, showfliers=False)

plt.suptitle("Boxplot of the numeric columns in the telecom dataset")
plt.tight_layout()
plt.show()



plt.figure(figsize=(25,12)) # width, height

for index in range(len(num_cols)):
    plt.subplot(2,3,index+1)
    sns.violinplot(y=df[num_cols[index]], x=df["Intl_Plan"], hue=df["Churn"], showfliers=False, split=True)

plt.suptitle("Boxplot of the numeric columns in the telecom dataset")
plt.tight_layout()
plt.show()





"""## Time Series Plot"""

# df['year'].unique()

df.head()

# df[df['model']==' 5 Series']

earthquakes = pd.read_csv("earthquakes.csv")

display(earthquakes.head())
display(earthquakes.tail())
print(earthquakes.info())
print(earthquakes.shape)

plt.figure(figsize=(20,8))
sns.lineplot(x="Year", y="earthquakes_per_year", data=earthquakes)

plt.title("Time-series plot of the earthquakes per year")
plt.xticks(np.arange(earthquakes["Year"].min(), earthquakes["Year"].max()+10, 10))
# plt.grid()
plt.show()

plt.figure(figsize=(20,8))
plt.plot(earthquakes["Year"], earthquakes["earthquakes_per_year"], linestyle="--")
plt.scatter(earthquakes["Year"], earthquakes["earthquakes_per_year"], marker="o", color="red")

plt.fill_between(earthquakes["Year"], 20, 30, color="b", alpha=0.25)

plt.title("Time-series plot of the earthquakes per year")
plt.xticks(np.arange(earthquakes["Year"].min(), earthquakes["Year"].max()+10, 10))
# plt.grid()

plt.xlabel("Year")
plt.ylabel("Number of earthquakes")
plt.show()



co2 = pd.read_csv("co2.csv")
co2['date'] = pd.to_datetime(co2['date'])

display(co2.head())
display(co2.tail())
display(co2.info())
print(co2.shape)

plt.figure(figsize=(20,8))
plt.plot(co2["date"], co2["CO2_ppm"])

plt.xlabel("Date")
plt.ylabel("CO2 (PPM)")
plt.yticks(np.arange(300, 500, 50))
plt.grid()
plt.show()


