import pandas as pd
import pymongo
import json

# Create a sample dataframe
df = pd.DataFrame({"name": ["Alice", "Bob", "Charlie"], "age": [25, 30, 35], "gender": ["F", "M", "M"]})

# Convert the dataframe to a JSON string
json_data = df.to_json(orient="records")

# Parse the JSON string to a list of dictionaries
records = json.loads(json_data)

# Connect to MongoDB and get the collection
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["test"]
collection = db["people"]

# Insert the records into the collection
collection.insert_many(records)
