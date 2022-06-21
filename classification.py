import re
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator

from pyspark.sql.functions import expr

spark = SparkSession.builder.appName('WorldDevelopment').getOrCreate()

data_folder = './WorldDevelopmentIndicators/'
df = spark.read.csv(data_folder + 'Indicators.csv',inferSchema=True,header=True)

df.printSchema()
df_romania = df.filter(df['CountryName']=='Romania')
df_romania.show()

'''
(Year, Indicator) -> (CountryName, Value)
'''

indicators = [
    'GDP per capita growth (annual %)',
    'Inflation, consumer prices (annual %)',
    'Real interest rate (%)',
    'Unemployment, total (% of total labor force)',
    'High-technology exports (% of manufactured exports)',
    'Labor force participation rate, total (% of total population ages 15+) (national estimate)',
    'GDP per capita growth (annual %)',
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

targetColumn = str(len(indicators) - 1)
#transformed_df = transformed_df.withColumn(targetColumn, transformed_df[targetColumn].cast('boolean'))

print("running transform", targetColumn)
transformed_df = transformed_df.withColumn(targetColumn, expr(f'CASE WHEN {targetColumn} < 0 THEN 0 ELSE 1 END'))               

transformed_df.show()

assembler = VectorAssembler(inputCols=['0',
 '1',
 '2',
 '3',
 '4',
 '5'],outputCol='features')

output = assembler.transform(transformed_df)

final_data = output.select('features','6')

train_churn,test_churn = final_data.randomSplit([0.7,0.3])

lr_churn = LogisticRegression(labelCol='6')

fitted_churn_model = lr_churn.fit(train_churn)

training_sum = fitted_churn_model.summary

training_sum.predictions.describe().show()

pred_and_labels = fitted_churn_model.evaluate(test_churn)

pred_and_labels.predictions.show()



churn_eval = BinaryClassificationEvaluator(rawPredictionCol='prediction',
                                           labelCol='6')

auc = churn_eval.evaluate(pred_and_labels.predictions)
print("AUC:", auc)

