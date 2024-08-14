#!/usr/bin/env python
# coding: utf-8

# In[61]:


#import section
import pymongo
from pymongo import MongoClient
from datetime import datetime
from pprint import pprint
import networkx as nx
import pandas as pd

# Creation of pyMongo connection object
client = MongoClient('localhost', 27017)
db = client['un1']
collection = db['un2']


# In[62]:


pipeline = [
    {
        "$group": {
            "_id": None,
            "minTimestamp": {"$min": "$timestamp_field"},
            "maxTimestamp": {"$max": "$timestamp_field"}
        }
    }
]


# In[86]:


result = list(collection.aggregate(pipeline))


# In[87]:


if result:
    min_timestamp = result[0]['minTimestamp']
    max_timestamp = result[0]['maxTimestamp']
    print("Minimum Timestamp:", min_timestamp)
    print("Maximum Timestamp:", max_timestamp)
else:
    print("No documents found in the collection.")


# In[88]:


from pymongo import MongoClient

# Connect to MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['un1']
collection = db['un2']

# Construct Aggregation Pipeline
pipeline = [
    {
        "$group": {
            "_id": "$country_field",
            "count": { "$sum": 1 }
        }
    },
    {
        "$group": {
            "_id": None,
            "total_countries": { "$sum": 1 }
        }
    }
]

# Execute Aggregation Pipeline
result = list(collection.aggregate(pipeline))

# Count Unique Countries
if result:
    total_countries = result[0]['total_countries']
    print("Total Unique Countries:", total_countries)
else:
    print("No documents found in the collection.")


# In[89]:


from pymongo import MongoClient

# Connect to MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['un1']
collection = db['un2']

# Define countries and year
countries = ['Tajikistan', 'Hungary', 'Germany', 'Ukraine']
year = 2015

# Construct Aggregation Pipeline
pipeline = [
    {
        "$match": {
            "country": {"$in": countries},
            "year": year
        }
    },
    {
        "$group": {
            "_id": "$country",
            "male_life_expectancy": {
                "$avg": {
                    "$cond": [{ "$eq": ["$sex", "Male"] }, "$value", None]
                }
            },
            "female_life_expectancy": {
                "$avg": {
                    "$cond": [{ "$eq": ["$sex", "Female"] }, "$value", None]
                }
            }
        }
    }
]

# Execute Aggregation Pipeline
results = list(collection.aggregate(pipeline))

# Print Results
for result in results:
    country = result['_id']
    male_life_expectancy = result['male_life_expectancy']
    female_life_expectancy = result['female_life_expectancy']
    print(f"Country: {country}")
    print(f"Male Life Expectancy: {male_life_expectancy}")
    print(f"Female Life Expectancy: {female_life_expectancy}")
    print()


# In[90]:


from pymongo import MongoClient

# Connect to MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['your_database_name']
collection = db['unece.jsondata']

# Define the year
year = 2010

# Construct Aggregation Pipeline
pipeline = [
    {
        "$match": {
            "year": year
        }
    },
    {
        "$group": {
            "_id": "$country",
            "total_population": { "$sum": "$value" }
        }
    },
    {
        "$sort": { "total_population": -1 }
    },
    {
        "$limit": 5
    }
]

# Execute Aggregation Pipeline
results = list(collection.aggregate(pipeline))

# Print Results
print("Top 5 Countries with the Highest Total Population in 2010:")
for index, result in enumerate(results, start=1):
    country = result['_id']
    total_population = result['total_population']
    print(f"{index}. {country}: {total_population}")


# In[91]:


# Print Results
print("Top 5 Countries with the Highest Total Population in 2010:")
for index, result in enumerate(results, start=1):
    country = result['_id']
    total_population = result['total_population']
    print(f"{index}. {country}: {total_population}")


# In[79]:


from pymongo import MongoClient

# Connect to MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['your_database_name']
collection = db['unece.jsondata']

# Define the year
year = 2010

# Construct Aggregation Pipeline
pipeline = [
    {
        "$match": {
            "year": year
        }
    },
    {
        "$group": {
            "_id": "$country",
            "total_population": { "$sum": "$value" }
        }
    },
    {
        "$sort": { "total_population": -1 }
    },
    {
        "$limit": 5
    }
]

# Execute Aggregation Pipeline
results = list(collection.aggregate(pipeline))

# Print Results
print("Top 5 Countries with the Highest Total Population in 2010:")
for index, result in enumerate(results, start=1):
    country = result['_id']
    total_population = result['total_population']
    print(f"{index}. {country}: {total_population}")


# In[69]:


from pymongo import MongoClient

# Connect to MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['your_database_name']
collection = db['unece.jsondata']

# Define the countries
countries = ['Israel', 'Kazakhstan', 'Kyrgyzstan', 'Latvia', 'Lithuania']

# Construct Aggregation Pipeline
pipeline = [
    {
        "$match": {
            "country": {"$in": countries},
            "age_group": "Adolescent"
        }
    },
    {
        "$group": {
            "_id": "$country",
            "average_rate": { "$avg": "$value" }
        }
    }
]

# Execute Aggregation Pipeline
results = list(collection.aggregate(pipeline))

# Print Results
print("Average Adolescent Rate for each country:")
for result in results:
    country = result['_id']
    average_rate = result['average_rate']
    print(f"{country}: {average_rate}")


# In[80]:


from pymongo import MongoClient

# Connect to MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['un1']
collection = db['un2']

# Define the year and threshold GDP per capita
year = 2000
gdp_per_capita_threshold = 100000

# Construct Aggregation Pipeline
pipeline = [
    {
        "$match": {
            "year": year,
            "indicator": "GDP per capita"
        }
    },
    {
        "$match": {
            "$expr": {
                "$gt": [ "$value", gdp_per_capita_threshold ]
            }
        }
    },
    {
        "$group": {
            "_id": "$country",
            "gdp_per_capita": { "$avg": "$value" }
        }
    }
]

# Execute Aggregation Pipeline
results = list(collection.aggregate(pipeline))

# Print Results
print("Countries with GDP per capita above $100,000 in 2000:")
for result in results:
    country = result['_id']
    gdp_per_capita = result['gdp_per_capita']
    print(f"{country}: {gdp_per_capita}")


# In[71]:


from pymongo import MongoClient

# Connect to MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['un1']
collection = db['un2']

# Define the years
years = [2000, 2016]

# Initialize dictionaries to store results
max_unemployment_rates = {}
min_unemployment_rates = {}

# Construct Aggregation Pipeline and find max/min unemployment rates for each year
for year in years:
    pipeline = [
        {
            "$match": {
                "year": year,
                "indicator": "Unemployment rate"
            }
        },
        {
            "$group": {
                "_id": "$country",
                "max_rate": { "$max": "$value" },
                "min_rate": { "$min": "$value" }
            }
        },
        {
            "$sort": { "max_rate": -1 }
        }
    ]
    
    # Execute Aggregation Pipeline
    results = list(collection.aggregate(pipeline))
    
    # Store max and min unemployment rates for the year
    max_unemployment_rates[year] = results[0] if results else None
    min_unemployment_rates[year] = results[-1] if results else None

# Print Results
print("Country with the highest and lowest unemployment rates:")
for year in years:
    max_result = max_unemployment_rates[year]
    min_result = min_unemployment_rates[year]
    
    max_country = max_result['_id'] if max_result else "No data available"
    max_rate = max_result['max_rate'] if max_result else "N/A"
    
    min_country = min_result['_id'] if min_result else "No data available"
    min_rate = min_result['min_rate'] if min_result else "N/A"
    
    print(f"Year: {year}")
    print(f"Highest Unemployment Rate: {max_country} - {max_rate}")
    print(f"Lowest Unemployment Rate: {min_country} - {min_rate}")
    print()


# In[81]:


from pymongo import MongoClient

# Connect to MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['un1']
collection = db['un2']

# Define the years and GDP per capita threshold
years = [2000, 2016]
gdp_per_capita_threshold = 10000

# Initialize list to store results
result_countries = []

# Construct Aggregation Pipeline
for year in years:
    pipeline = [
        {
            "$match": {
                "year": year,
                "indicator": "GDP per capita",
                "value": { "$gt": gdp_per_capita_threshold }
            }
        },
        {
            "$project": {
                "_id": 0,
                "country": "$country",
                "gdp_per_capita": "$value"
            }
        }
    ]
    
    # Execute Aggregation Pipeline
    results = list(collection.aggregate(pipeline))
    
    # Append results to the list
    result_countries.extend(results)

# Count the total number of countries
total_countries = len(result_countries)

# Print Results
print(f"Total Number of Countries: {total_countries}")
print("Countries with GDP per capita above $10,000:")
for result in result_countries:
    country = result['country']
    gdp_per_capita = result['gdp_per_capita']
    print(f"Country: {country}, GDP per capita: {gdp_per_capita}")


# In[73]:


from pymongo import MongoClient

# Connect to MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['un1']
collection = db['un2']

# Define the criteria
fertility_rate_threshold = 2
cpi_growth_rate_threshold = 3

# Construct Aggregation Pipeline
pipeline = [
    {
        "$match": {
            "indicator": "Total fertility rate"
        }
    },
    {
        "$group": {
            "_id": "$country",
            "avg_fertility_rate": { "$avg": "$value" }
        }
    },
    {
        "$match": {
            "avg_fertility_rate": { "$gt": fertility_rate_threshold }
        }
    },
    {
        "$lookup": {
            "from": "unece.jsondata",
            "let": { "country_name": "$_id" },
            "pipeline": [
                {
                    "$match": {
                        "$expr": {
                            "$and": [
                                { "$eq": ["$country", "$$country_name"] },
                                { "$eq": ["$indicator", "Consumer price index growth rate"] }
                            ]
                        }
                    }
                }
            ],
            "as": "cpi_growth"
        }
    },
    {
        "$unwind": "$cpi_growth"
    },
    {
        "$match": {
            "cpi_growth.value": { "$gt": cpi_growth_rate_threshold }
        }
    },
    {
        "$project": {
            "_id": 0,
            "country": "$_id",
            "avg_fertility_rate": 1,
            "cpi_growth_rate": "$cpi_growth.value"
        }
    }
]

# Execute Aggregation Pipeline
results = list(collection.aggregate(pipeline))

# Print Results
print("Countries with Total Fertility Rate > 2 and CPI Growth Rate > 3%:")
for result in results:
    country = result['country']
    avg_fertility_rate = result['avg_fertility_rate']
    cpi_growth_rate = result['cpi_growth_rate']
    print(f"Country: {country}, Avg. Fertility Rate: {avg_fertility_rate}, CPI Growth Rate: {cpi_growth_rate}")


# In[74]:


from pymongo import MongoClient

# Connect to MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['un1']
collection = db['un2']

# Define the criteria
fertility_rate_threshold = 2
cpi_growth_rate_threshold = 3

# Construct Aggregation Pipeline
pipeline = [
    {
        "$match": {
            "indicator": "Total fertility rate"
        }
    },
    {
        "$group": {
            "_id": "$country",
            "avg_fertility_rate": { "$avg": "$value" }
        }
    },
    {
        "$match": {
            "avg_fertility_rate": { "$gt": fertility_rate_threshold }
        }
    },
    {
        "$lookup": {
            "from": "unece.jsondata",
            "let": { "country_name": "$_id" },
            "pipeline": [
                {
                    "$match": {
                        "$expr": {
                            "$and": [
                                { "$eq": ["$country", "$$country_name"] },
                                { "$eq": ["$indicator", "Consumer price index growth rate"] }
                            ]
                        }
                    }
                }
            ],
            "as": "cpi_growth"
        }
    },
    {
        "$unwind": "$cpi_growth"
    },
    {
        "$match": {
            "cpi_growth.value": { "$gt": cpi_growth_rate_threshold }
        }
    },
    {
        "$project": {
            "_id": 0,
            "country": "$_id",
            "avg_fertility_rate": 1,
            "cpi_growth_rate": "$cpi_growth.value"
        }
    }
]

# Execute Aggregation Pipeline
results = list(collection.aggregate(pipeline))

# Print Results
print("Countries with Total Fertility Rate > 2 and CPI Growth Rate > 3%:")
for result in results:
    country = result['country']
    avg_fertility_rate = result['avg_fertility_rate']
    cpi_growth_rate = result['cpi_growth_rate']
    print(f"Country: {country}, Avg. Fertility Rate: {avg_fertility_rate}, CPI Growth Rate: {cpi_growth_rate}")


# In[75]:


from pymongo import MongoClient

# Connect to MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['un1']
collection = db['un2']

# Construct Aggregation Pipeline
pipeline = [
    {
        "$match": {
            "indicator": "Population, total"
        }
    },
    {
        "$group": {
            "_id": "$country",
            "total_population": { "$sum": "$value" }
        }
    },
    {
        "$sort": { "total_population": -1 }
    }
]

# Execute Aggregation Pipeline
results = list(collection.aggregate(pipeline))

# Print Results
print("Total Population by Country (Descending Order):")
for result in results:
    country = result['_id']
    total_population = result['total_population']
   


# In[103]:


print(f"Country: {country}, Total Population: {total_population}")


# In[60]:


# Task a: Summarize the dataset
total_documents = collection.count_documents({})
print("Total documents in the collection:", total_documents)

# Fetching the first document to understand its structure
sample_document = collection.find_one()
print("Sample document:", sample_document)

# Getting the list of keys (fields) present in the documents
keys = sample_document.keys()
print("Fields present in the documents:", keys)

# Displaying data types of each field in the sample document
for key, value in sample_document.items():
    print(f"Field: {key}, Data Type: {type(value)}")


# In[82]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Retrieve all documents from the collection
documents = list(collection.find())

# Convert documents to DataFrame for analysis
df = pd.DataFrame('un1')

# Identify numeric fields
numeric_fields = df.select_dtypes(include=[np.number]).columns.tolist()

# Display numeric fields
print("Numeric Fields:", numeric_fields)

# Describe numeric fields
numeric_data_description = df[numeric_fields].describe()
print("\nDescriptive statistics for numeric fields:")
print(numeric_data_description)

# Visualize distribution of numeric fields
for field in numeric_fields:
    plt.figure(figsize=(8, 6))
    plt.hist(df[field], bins=20, color='skyblue', edgecolor='black')
    plt.title(f'Distribution of {field}')
    plt.xlabel(field)
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()


# In[34]:


a.	Find the timeframe of Your MongoDB Collection

pipeline = [
    {
        "$group": {
            "_id": null,
            "earliestDate": { "$min": "$Year" },
            "latestDate": { "$max": "$Year" }
        }
    }
];

result = db.unece.aggregate(pipeline).toArray();

// Extract time frame from the result
if (result.length > 0) {
    print("Earliest Date:", result[0].earliestDate);
    print("Latest Date:", result[0].latestDate);
} else {
    print("Collection is empty or does not contain a timestamp field.");
}


# In[35]:


# a. Summarize the dataset
total_documents = collection.count_documents({})
print("Total number of documents:", total_documents)
print("Number of features:", len(collection.find_one({}).keys()))

# Convert MongoDB collection to Pandas DataFrame
df = pd.DataFrame(list(collection.find()))

# a. Summarize the dataset
print("Dataset Summary:")
print(df.info())


# In[83]:


a.	Find the timeframe of Your MongoDB Collection

pipeline = [
    {
        "$group": {
            "_id": null,
            "earliestDate": { "$min": "$Year" },
            "latestDate": { "$max": "$Year" }
        }
    }
];

result = db.unece.aggregate(pipeline).toArray();

// Extract time frame from the result
if (result.length > 0) {
    print("Earliest Date:", result[0].earliestDate);
    print("Latest Date:", result[0].latestDate);
} else {
    print("Collection is empty or does not contain a timestamp field.");
}

b.	Count the number of all unique countries enlisted in your MongoDB Collection

// Find all unique countries
var unique_countries = db.unece.distinct("Country");

// Count the number of unique countries
var num_unique_countries = unique_countries.length;

print("Number of unique countries:", num_unique_countries);


c.	Find the life expectancy at birth for both men and women in Tajikistan, hungary, Germany, Ukraine in the year 2015
// Define the countries and year
var countries = ["Tajikistan", "Hungary", "Germany", "Ukraine"];
var year = "2015";

// Perform aggregation to find life expectancy at birth
var pipeline = [
    // Filter documents for the specified countries and year
    { $match: { Country: { $in: countries }, Year: year } },
    // Project only the relevant fields
    { $project: { Country: 1, "Life expectancy at birth, women": 1, "Life expectancy at birth, men": 1 } }
];

// Execute the aggregation pipeline
var results = db.unece.aggregate(pipeline).toArray();

// Print the results
results.forEach(function(result) {
    var country = result.Country;
    var life_expectancy_women = result["Life expectancy at birth, women"] || "N/A";
    var life_expectancy_men = result["Life expectancy at birth, men"] || "N/A";
    print("Country: " + country + ", Life Expectancy at Birth (Women): " + life_expectancy_women + ", Life Expectancy at Birth (Men): " + life_expectancy_men);
});

d.	Find the top 5 countries with the highest total population in the year 2010

 // Query d. Find the top 5 countries with the highest total population in the year 2010
var pipeline_d = [
    { $match: { Year: "2010" } },
    { $sort: { "Total population": -1 } },
    { $limit: 5 },
    { $project: { _id: 0, Country: 1, Total_population: 1 } }
];
var result_d = db.unece.aggregate(pipeline_d).toArray();
print("Top 5 countries with the highest total population in 2010:");
result_d.forEach(function(doc) {
    printjson(doc);
});

e.	Calculate the average adolescent fertility rate for ‘Isral’. ‘Kazakhstan’, ‘Kyrgyzstan’, ‘Latvia’, ‘Lituania’
// Query e. Calculate the average adolescent fertility rate for the specified countries
var pipeline_e = [
    { $match: { Country: { $in: ["Israel", "Italy", "Kazakhstan", "Kyrgyzstan", "Latvia", "Lithuania"] } } },
    { $group: { _id: null, avg_fertility_rate: { $avg: "$Adolescent fertility rate" } } }
];
var result_e = db.unece.aggregate(pipeline_e).toArray();
print("Average adolescent fertility rate for the specified countries:");
print(result_e[0].avg_fertility_rate);

F. Countries with GDP per capita above $10,000 in the year 2000
// Define the pipeline to find countries with GDP per capita above $10,000 in the year 2000
var pipeline_f = [
    { $match: { Year: "2000", "GDP per capita at current prices and PPPs, US$": { $gt: 10000 } } },
    { $project: { _id: 0, Country: 1 } }
];

// Execute the aggregation pipeline
var result_f = db.unece.aggregate(pipeline_f).toArray();

// Print the result
print("Countries with a GDP per capita above $10,000 in the year 2000:");
result_f.forEach(function(doc) {
    printjson(doc);
});

G. Country with the lowest and highest unemployment rates in the year 2000
var pipeline_g = [
    { $match: { Year: { $in: ["2000", "2016"] } } },
    { $group: { _id: { Year: "$Year", Country: "$Country" }, avg_unemployment_rate: { $avg: "$Unemployment rate" } } },
    { $sort: { avg_unemployment_rate: 1 } }
];

// Execute the aggregation pipeline
var result_g = db.unece.aggregate(pipeline_g).toArray();

// Print the country with the lowest unemployment rate in the year 2000
print("Country with the lowest unemployment rate in the year 2000:", result_g[0]._id.Country);

// Print the country with the highest unemployment rate in the year 2000
print("Country with the highest unemployment rate in the year 2000:", result_g[result_g.length - 1]._id.Country);

h. Count the total number of countries with GDP per capita above $10,000 in 2000 and 2016
var pipeline_h = [
    { $match: { Year: { $in: ["2000", "2016"] }, "GDP per capita at current prices and PPPs, US$": { $gt: 10000 } } },
    { $project: { _id: 0, Country: 1, GDP_per_capita: "$GDP per capita at current prices and PPPs, US$" } },
    { $group: { _id: "$Country", GDP_per_capita: { $first: "$GDP_per_capita" } } },
    { $count: "total_countries" }
];

// Execute the aggregation pipeline
var result_h = db.unece.aggregate(pipeline_h).toArray();

// Print the total number of countries with GDP per capita above $10,000 in 2000 and 2016
print("Total number of countries with GDP per capita above $10,000 in 2000 and 2016:", result_h[0].total_countries);



i.	find countries with a total fertility rate greater than 2 and a consumer price index growth rate greater than 3%
var pipeline_i = [
    { $match: { "Total fertility rate": { $gt: 2 }, "Consumer price index, growth rate": { $gt: 0.03 } } },
    { $project: { _id: 0, Country: 1 } }
];

// Execute the aggregation pipeline
var result_i = db.unece.aggregate(pipeline_i).toArray();

// Print the countries
print("Countries with a total fertility rate greater than 2 and a consumer price index growth rate greater than 3%:");
result_i.forEach(function(doc) {
    printjson(doc);
});

J. Calculate the total population for each country and sort in descending order
var pipeline_j = [
    { $group: { _id: "$Country", total_population: { $sum: "$Total population" } } },
    { $sort: { total_population: -1 } }
];

// Execute the aggregation pipeline
var result_j = db.unece.aggregate(pipeline_j).toArray();

// Print the result
print("Total population for each country sorted in descending order:");
result_j.forEach(function(doc) {
    printjson(doc);
});


# In[84]:


J. Calculate the total population for each country and sort in descending order
var pipeline_j = [
    { $group: { _id: "$Country", total_population: { $sum: "$Total population" } } },
    { $sort: { total_population: -1 } }
];

// Execute the aggregation pipeline
var result_j = db.unece.aggregate(pipeline_j).toArray();

// Print the result
print("Total population for each country sorted in descending order:");
result_j.forEach(function(doc) {
    printjson(doc);
});


# In[92]:


from pymongo import MongoClient

# Connect to MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['un1']
collection = db['un2']

# Define the years and GDP per capita threshold
years = [2000, 2016]
gdp_per_capita_threshold = 10000

# Initialize list to store results
result_countries = []

# Construct Aggregation Pipeline
for year in years:
    pipeline = [
        {
            "$match": {
                "year": year,
                "indicator": "GDP per capita",
                "value": { "$gt": gdp_per_capita_threshold }
            }
        },
        {
            "$project": {
                "_id": 0,
                "country": "$country",
                "gdp_per_capita": "$value"
            }
        }
    ]
    
    # Execute Aggregation Pipeline
    results = list(collection.aggregate(pipeline))
    
    # Append results to the list
    result_countries.extend(results)

# Count the total number of countries
total_countries = len(result_countries)

# Print Results
print(f"Total Number of Countries: {total_countries}")
print("Countries with GDP per capita above $10,000:")
for result in result_countries:
    country = result['country']
    gdp_per_capita = result['gdp_per_capita']
    print(f"Country: {country}, GDP per capita: {gdp_per_capita}")


# In[93]:


from pymongo import MongoClient

# Connect to MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['un1']
collection = db['un2']

# Define the criteria
fertility_rate_threshold = 2
cpi_growth_rate_threshold = 3

# Construct Aggregation Pipeline
pipeline = [
    {
        "$match": {
            "indicator": "Total fertility rate"
        }
    },
    {
        "$group": {
            "_id": "$country",
            "avg_fertility_rate": { "$avg": "$value" }
        }
    },
    {
        "$match": {
            "avg_fertility_rate": { "$gt": fertility_rate_threshold }
        }
    },
    {
        "$lookup": {
            "from": "unece.jsondata",
            "let": { "country_name": "$_id" },
            "pipeline": [
                {
                    "$match": {
                        "$expr": {
                            "$and": [
                                { "$eq": ["$country", "$$country_name"] },
                                { "$eq": ["$indicator", "Consumer price index growth rate"] }
                            ]
                        }
                    }
                }
            ],
            "as": "cpi_growth"
        }
    },
    {
        "$unwind": "$cpi_growth"
    },
    {
        "$match": {
            "cpi_growth.value": { "$gt": cpi_growth_rate_threshold }
        }
    },
    {
        "$project": {
            "_id": 0,
            "country": "$_id",
            "avg_fertility_rate": 1,
            "cpi_growth_rate": "$cpi_growth.value"
        }
    }
]

# Execute Aggregation Pipeline
results = list(collection.aggregate(pipeline))



# In[94]:


from pymongo import MongoClient

# Connect to MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['un1']
collection = db['un2']
# Print Results
print("Countries with Total Fertility Rate > 2 and CPI Growth Rate > 3%:")
for result in results:
    country = result['country']
    avg_fertility_rate = result['avg_fertility_rate']
    cpi_growth_rate = result['cpi_growth_rate']
    print(f"Country: {country}, Avg. Fertility Rate: {avg_fertility_rate}, CPI Growth Rate: {cpi_growth_rate}")


# In[95]:


from pymongo import MongoClient

# Connect to MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['un1']
collection = db['un2']

# Construct Aggregation Pipeline
pipeline = [
    {
        "$match": {
            "indicator": "Population, total"
        }
    },
    {
        "$group": {
            "_id": "$country",
            "total_population": { "$sum": "$value" }
        }
    },
    {
        "$sort": { "total_population": -1 }
    }
]

# Execute Aggregation Pipeline
results = list(collection.aggregate(pipeline))

# Print Results
print("Total Population by Country (Descending Order):")
for result in results:
    country = result['_id']
    total_population = result['total_population']
    print(f"Country: {country}, Total Population: {total_population}")
    


# In[96]:


print(f"Country: {country}, Total Population: {total_population}")
    


# In[98]:


var pipeline_e = [
    { $match: { Country: { $in: ["Israel", "Italy", "Kazakhstan", "Kyrgyzstan", "Latvia", "Lithuania"] } } },
    { $group: { _id: null, avg_fertility_rate: { $avg: "$Adolescent fertility rate" } } }
];
var result_e = db.unece.aggregate(pipeline_e).toArray();
print("Average adolescent fertility rate for the specified countries:");
print(result_e[0].avg_fertility_rate);


# In[4]:


#import section
import pymongo
from pymongo import MongoClient
from datetime import datetime
from pprint import pprint
import networkx as nx
import pandas as pd

# Creation of pyMongo connection object
client = MongoClient('localhost', 27017)
db = client['un1']
collection = db['un2']



pipeline = [
    {"$match": {"$or": [{"year": 2000}, {"year": 2016}]}},
    {"$group": {"_id": {"country": "$country", "year": "$year"}, "gdp_per_capita": {"$avg": "$gdp_per_capita"}}},
    {"$match": {"gdp_per_capita": {"$gt": 10000}}},
    {"$project": {"_id": 0, "country": "$_id.country", "gdp_per_capita": "$gdp_per_capita"}}
]

result = list(collection.aggregate(pipeline))
total_countries = len(result)
pprint(result)
print("Total number of countries with GDP per capita above $10,000:", total_countries)


# In[3]:


#import section
import pymongo
from pymongo import MongoClient
from datetime import datetime
from pprint import pprint
import networkx as nx
import pandas as pd

# Creation of pyMongo connection object
client = MongoClient('localhost', 27017)
db = client['un1']
collection = db['un2']




pipeline = [
    {"$match": {"$and": [{"total_fertility_rate": {"$gt": 2}}, {"consumer_price_index_growth": {"$gt": 0.03}}]}},
    {"$project": {"_id": 0, "country": 1}}
]

result = list(collection.aggregate(pipeline))
total_countries = len(result)
pprint(result)
print("Total number of countries:", total_countries)


# In[5]:


#import section
import pymongo
from pymongo import MongoClient
from datetime import datetime
from pprint import pprint
import networkx as nx
import pandas as pd

# Creation of pyMongo connection object
client = MongoClient('localhost', 27017)
db = client['un1']
collection = db['un2']


# In[6]:


from pymongo import MongoClient

# Connect to MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['un1']
collection = db['un2']

# Construct Aggregation Pipeline
pipeline = [
    {
        "$match": {
            "indicator": "Population, total"
        }
    },
    {
        "$group": {
            "_id": "$country",
            "total_population": { "$sum": "$value" }
        }
    },
    {
        "$sort": { "total_population": -1 }
    }
]

# Execute Aggregation Pipeline
results = collection.aggregate(pipeline)

# Print Results
print("Total Population by Country (Descending Order):")
for result in results:



# In[3]:


country = result['_id']
total_population = result['total_population']
print(f"Country: {country}, Total Population: {total_population}")



# In[4]:


Print Results
print("Total Population by Country (Descending Order):")
for result in results:


# In[ ]:


#import section
import pymongo
from pymongo import MongoClient
from datetime import datetime
from pprint import pprint
import networkx as nx
import pandas as pd

# Creation of pyMongo connection object
client = MongoClient('localhost', 27017)
db = client['un1']
collection = db['un2']


# In[9]:


from pymongo import MongoClient

# Connect to MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['un1']
collection = db['un2']

# Construct Aggregation Pipeline
pipeline = [
    {
        "$match": {
            "indicator": "Population, total"
        }
    },
    {
        "$group": {
            "_id": "$country",
            "total_population": { "$sum": "$value" }
        }
    },
    {
        "$sort": { "total_population": -1 }
    }
]

# Execute Aggregation Pipeline
results = collection.aggregate(pipeline)

)


# In[10]:


# Print Results
print("Total Population by Country (Descending Order):")
for result in results:
    country = result['_id']
    total_population = result['total_population']
    print(f"Country: {country}, Total Population: {total_population}"


# In[11]:


# Print Results
print("Total Population by Country (Descending Order):")
for result in results:
    country = result['_id']
    total_population = result['total_population']
    print(f"Country: {country}, Total Population: {total_population}")


# In[12]:


# Print Results
print("Total Population by Country (Descending Order):")
for result in results:
    country = result['_id']
    total_population = result['total_population']
    print(f"Country: {country}, Total Population: {total_population}")


# In[ ]:




