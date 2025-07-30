from pyspark.sql import SparkSession

# Initialize SparkSession
spark = SparkSession.builder.appName("IC_Keystroke_Dynamics").getOrCreate()

# Path to the directory containing your Parquet files
directory_path = "results\\one_class_st_user_hp_tuning_cmu\\2025-07-22_13-52-42"

# PySpark reads all Parquet files in the directory into a single DataFrame
df = spark.read.parquet(directory_path)

df.printSchema()
df.show()