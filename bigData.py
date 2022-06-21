import re
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression

spark = SparkSession.builder.appName('WorldDevelopment').getOrCreate()

data_folder = './WorldDevelopmentIndicators/'
df = spark.read.csv(data_folder + 'Indicators.csv',inferSchema=True,header=True)

df.printSchema()
df_romania = df.filter(df['CountryName']=='Romania')
df_romania.show()

# Collect the y variable and x variables 
y = ['CO2 emissions \(metric tons per capita\)']

x = ['GDP growth (annual %)', 
     'GDP per capita growth (annual %)', 
     'Inflation, consumer prices (annual %)', 
     'Population growth (annual %)',
     'Fossil fuel energy consumption (% of total)', 
     'Renewable energy consumption (% of total final energy consumption)',
     'Alternative and nuclear energy (% of total energy use)', 
     'Forest area (% of land area)']

# for i in range(len(x)):
#     filter_data = df_romania.filter(df_romania['IndicatorName']==x[i])
#     filter_data.show()
#     x_years = filter_data.select('Year')
#     y_values = filter_data.select('Value')
#     plt.xlabel('Years')
#     plt.ylabel(x[i])
#     plt.title(f'{x[i]} over time for Romania')
    
#     #plot function 
#     plt.plot(x_years.collect(), y_values.collect())
#     plt.show()

'''
(Year, Indicator) -> (CountryName, Value)
'''

indicators = [
    'Population density (people per sq. km of land area)',
    'GDP per capita growth (annual %)',
    'Life expectancy at birth, total (years)',
    'Population ages 65 and above (% of total)',
    'Surface area (sq. km)',
    'Adolescent fertility rate (births per 1,000 women ages 15-19)',
    'Fertility rate, total (births per woman)'
]

transformed_df = df_romania \
    .filter(df_romania.IndicatorName==indicators[0]) \
    .select('CountryName', 'Year', 'Value') \
    .withColumnRenamed('Value', '0')

for i in range(1, len(indicators)):
    new_df = df_romania \
        .filter(df_romania.IndicatorName==indicators[i]) \
        .select('CountryName', 'Year', 'Value') \
        .withColumnRenamed('Value', str(i))
    
    transformed_df = transformed_df.join(new_df, ['CountryName', 'Year'])


indexed = StringIndexer(inputCol='CountryName', outputCol='fertility_cat').fit(transformed_df).transform(transformed_df)
indexed.head(5)

assembler = VectorAssembler(
  inputCols=['0', '1', '2', '3', '4', '5', '6'],
    outputCol='features')

# assembler = VectorAssembler(
#   inputCols=[
#     'Population density (people per sq. km of land area)',
#     'GDP per capita growth (annual %)',
#     'Life expectancy at birth, total (years)',
#     'Population ages 65 and above (% of total)',
#     'Surface area (sq. km)',
#     'Adolescent fertility rate (births per 1,000 women ages 15-19)',
#     'Fertility rate, total (births per woman)',
#     'fertility_cat'
#     ],
#     outputCol='features')

output = assembler.transform(indexed)

final_data = output.select('features', '6')

train_data,test_data = final_data.randomSplit([0.7,0.3])

lr = LinearRegression(labelCol='6')

lrModel = lr.fit(train_data)

print("Coefficients: {} Intercept: {}".format(lrModel.coefficients,lrModel.intercept))

test_results = lrModel.evaluate(test_data)

# Afișați informații obținute în urma evaluării
print("RMSE: {}".format(test_results.rootMeanSquaredError))
print("MSE: {}".format(test_results.meanSquaredError))
print("R2: {}".format(test_results.r2))

unlabeled_data=test_data.select("features")
predictions = lrModel.transform(unlabeled_data)
predictions.show()

test_data.show()