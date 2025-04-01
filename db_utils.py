from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError
from config import DB_CONFIG

class MongoDBClient:
    def __init__(self):
        self.client = None
        self.db = None
        self.collection = None

    def connect(self):
        try:
            self.client = MongoClient(DB_CONFIG["uri"], serverSelectionTimeoutMS=DB_CONFIG["timeout_ms"])
            self.client.admin.command('ping')
            self.db = self.client[DB_CONFIG["name"]]
            self.collection = self.db[DB_CONFIG["collection"]]
            print("Connected to MongoDB successfully!")
        except ServerSelectionTimeoutError as e:
            print(f"Connection error: {e}")
            raise
        return self.collection

    def get_collection(self):
        if not self.collection:
            self.connect()
        return self.collection

    def close(self):
        if self.client:
            self.client.close()
            print("MongoDB connection closed.")