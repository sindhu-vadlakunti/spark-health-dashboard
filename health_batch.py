from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, round, row_number, lit, avg, sum as Fsum, floor, count
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans
from tabulate import tabulate

# --- Initialize Spark Session ---
spark = SparkSession.builder.appName("🏥 Family Health Analytics").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

print("\n" + "="*90)
print("=== 🏥 FAMILY HEALTH ANALYTICS BATCH PROCESS STARTED ===")
print("="*90)

# --- Load Dataset ---
data_path = "file:///home/sindhu/bigdata/datasets/public_health.csv"
df = spark.read.option("header", True).csv(data_path, inferSchema=True)
print(f"\n✅ Loaded dataset successfully: {df.count()} records\n")

# --- Convert columns to numeric types ---
numeric_cols = ["Immunity_Level", "Vaccination_Status", "Chronic_Conditions", "Temperature", "AQI", "Humidity"]
for c in numeric_cols:
    df = df.withColumn(c, col(c).cast("double"))

# --- Fill missing values ---
df = df.fillna({
    "Chronic_Conditions": 0,
    "Immunity_Level": 0.5,
    "Vaccination_Status": 0.5,
    "Reported_Symptoms": "",
    "AQI": 50,
    "Temperature": 36.5,
    "Humidity": 50
})

# --- Health Calculations ---
df = df.withColumn("Chronic_Condition_Risk", when(col("Chronic_Conditions") > 0, 1).otherwise(0)) \
       .withColumn("Symptom_Severity", when(col("Reported_Symptoms") != "", 1).otherwise(0)) \
       .withColumn("Wellness_Score", round(
           0.4 * (1 - col("Chronic_Condition_Risk")) +
           0.3 * col("Immunity_Level") +
           0.2 * col("Vaccination_Status") +
           0.1 * (1 - col("Symptom_Severity")), 2)) \
       .withColumn("Risk_Level", when(col("Wellness_Score") >= 0.7, "Low")
                   .when((col("Wellness_Score") >= 0.4) & (col("Wellness_Score") < 0.7), "Medium")
                   .otherwise("High")) \
       .withColumn("Predicted_Hospitalization",
                   when((col("Wellness_Score") < 0.5) | (col("Chronic_Condition_Risk") == 1), "Yes").otherwise("No")) \
       .withColumn("Environmental_Alert",
                   when(col("AQI") > 100, "High Risk Environment")
                   .when(col("AQI") > 50, "Moderate Risk Environment")
                   .otherwise("Safe Environment"))

# --- Composite Risk & Recommendations ---
df = df.withColumn("Composite_Risk_Score", round(
           0.5 * (1 - col("Wellness_Score")) +
           0.3 * col("Chronic_Condition_Risk") +
           0.2 * (col("Environmental_Alert") == "High Risk Environment").cast("double"), 2)) \
       .withColumn("Composite_Risk_Level", when(col("Composite_Risk_Score") >= 0.7, "High")
                   .when((col("Composite_Risk_Score") >= 0.4) & (col("Composite_Risk_Score") < 0.7), "Medium")
                   .otherwise("Low")) \
       .withColumn("Risk_Percentile", round(col("Composite_Risk_Score") * 100, 2)) \
       .withColumn("Recommendation", when(col("Composite_Risk_Level") == "High", "[RED] Immediate medical checkup recommended")
                   .when(col("Composite_Risk_Level") == "Medium", "[YELLOW] Monitor health regularly")
                   .otherwise("[GREEN] Maintain current health routine"))

# --- Family ID Assignment ---
window_spec = Window.orderBy(lit(1))
df = df.withColumn("row_num", row_number().over(window_spec))
df = df.withColumn("Family_ID", floor((col("row_num") - 1) / 5) + 1).drop("row_num")

# --- Family-level Aggregation ---
family_df = df.groupBy("Family_ID").agg(
    round(avg("Wellness_Score"), 2).alias("Family_Wellness_Score"),
    round(avg("Immunity_Level"), 2).alias("Avg_Immunity_Level"),
    round(avg("Vaccination_Status"), 2).alias("Avg_Vaccination_Status"),
    Fsum("Chronic_Condition_Risk").alias("Total_Chronic_Conditions"),
    Fsum((col("Predicted_Hospitalization") == "Yes").cast("double")).alias("Hospitalization_Risk_Count"),
    Fsum((col("Environmental_Alert") == "High Risk Environment").cast("double")).alias("High_Env_Risk_Count"),
    round(avg("Composite_Risk_Score"), 2).alias("Avg_Composite_Risk")
)

# --- KMeans Clustering ---
assembler = VectorAssembler(
    inputCols=["Family_Wellness_Score", "Avg_Composite_Risk", "Avg_Immunity_Level", "Total_Chronic_Conditions"],
    outputCol="features"
)
feature_df = assembler.transform(family_df)
kmeans = KMeans(k=4, seed=42, featuresCol="features", predictionCol="Cluster")
model = kmeans.fit(feature_df)
clustered_family_df = model.transform(feature_df)
df = df.join(clustered_family_df.select("Family_ID", "Cluster"), on="Family_ID", how="left")

# --- Export Final Dataset ---
output_path = "file:///home/sindhu/bigdata/output/health.csv"
df.write.csv(output_path, header=True, mode="overwrite")
print(f"\n✅ Data exported successfully to {output_path}\n")

# --- Display Results Beautifully ---
print("📊 FAMILY CLUSTER SUMMARY:")
family_table = clustered_family_df.select("Family_ID", "Family_Wellness_Score", "Avg_Composite_Risk", "Avg_Immunity_Level",
                                           "Total_Chronic_Conditions", "Hospitalization_Risk_Count", "Cluster") \
                                  .limit(20).toPandas()
print(tabulate(family_table, headers='keys', tablefmt='fancy_grid', showindex=False))

print("\n📈 RISK LEVEL COUNTS:")
risk_counts = df.groupBy("Risk_Level").agg(count("*").alias("Count")).orderBy("Risk_Level").toPandas()
print(tabulate(risk_counts, headers='keys', tablefmt='fancy_grid', showindex=False))

print("\n🧠 COMPOSITE RISK LEVEL SUMMARY:")
comp_counts = df.groupBy("Composite_Risk_Level").agg(count("*").alias("Count")).orderBy("Composite_Risk_Level").toPandas()
print(tabulate(comp_counts, headers='keys', tablefmt='fancy_grid', showindex=False))

print("\n" + "="*90)
print("✅ BATCH PROCESS COMPLETED SUCCESSFULLY 🎯")
print("="*90)

spark.stop()
