from pymongo import MongoClient
import pandas as pd
client = MongoClient('mongodb://localhost:27017/')
db=client.myapp
collection = db.app

url="E:/Myproject_am/static/json/fastfood_chain_data/test5.json"
df = pd.read_json(url)
#df2=pd.DataFrame(df)
df2=pd.DataFrame(df)
post_id = db.app.insert_many(df2.to_dict('records'))
#print(post_id.inserted_ids)
