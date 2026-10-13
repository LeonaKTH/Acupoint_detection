"""Per-user result history; SQLite locally, MongoDB when MONGODB_URI is set."""
import json
import sqlite3
from pathlib import Path
from datetime import datetime, timezone
from uuid import uuid4


class HistoryStore:
    def __init__(self, sqlite_path="outputs/history.db", mongo_uri=None):
        self.sqlite_path = Path(sqlite_path)
        self.mongo_uri = mongo_uri
        self._mongo = None

    def mongo(self):
        if self._mongo is None:
            from pymongo import MongoClient
            self._mongo = MongoClient(self.mongo_uri, serverSelectionTimeoutMS=3000)
            collection = self._mongo["acupoint"]["history"]
            collection.create_index([("user_id", 1), ("created_at", -1)])
        return self._mongo["acupoint"]["history"]

    def connect(self):
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.sqlite_path)
        connection.execute("CREATE TABLE IF NOT EXISTS history (id TEXT PRIMARY KEY, user_id TEXT, created_at TEXT, payload TEXT)")
        connection.execute("CREATE INDEX IF NOT EXISTS history_user_date ON history(user_id, created_at)")
        return connection

    def add(self, user_id, result):
        record = {"id": uuid4().hex, "user_id": user_id,
                  "created_at": datetime.now(timezone.utc).isoformat(), "result": result}
        if self.mongo_uri:
            self.mongo().insert_one(dict(record))
        else:
            with self.connect() as connection:
                connection.execute("INSERT INTO history VALUES (?,?,?,?)",
                                   (record["id"],user_id,record["created_at"],json.dumps(result,ensure_ascii=False)))
        return record["id"]

    def list(self, user_id, limit=20):
        limit = max(1,min(100,limit))
        if self.mongo_uri:
            return list(self.mongo().find({"user_id":user_id},{"_id":0}).sort("created_at",-1).limit(limit))
        with self.connect() as connection:
            rows = connection.execute("SELECT id,created_at,payload FROM history WHERE user_id=? ORDER BY created_at DESC LIMIT ?",
                                      (user_id,limit)).fetchall()
        return [{"id":r[0],"created_at":r[1],"result":json.loads(r[2])} for r in rows]

    def set_profile(self,user_id,profile):
        if self.mongo_uri:
            self.mongo().database["profiles"].update_one({"user_id":user_id},{"$set":{"profile":profile}},upsert=True)
        else:
            with self.connect() as connection:
                connection.execute("CREATE TABLE IF NOT EXISTS profiles (user_id TEXT PRIMARY KEY, payload TEXT)")
                connection.execute("INSERT INTO profiles VALUES (?,?) ON CONFLICT(user_id) DO UPDATE SET payload=excluded.payload",
                                   (user_id,json.dumps(profile,ensure_ascii=False)))
        return profile

    def get_profile(self,user_id):
        if self.mongo_uri:
            record=self.mongo().database["profiles"].find_one({"user_id":user_id})
            return record["profile"] if record else {}
        with self.connect() as connection:
            connection.execute("CREATE TABLE IF NOT EXISTS profiles (user_id TEXT PRIMARY KEY, payload TEXT)")
            row=connection.execute("SELECT payload FROM profiles WHERE user_id=?",(user_id,)).fetchone()
        return json.loads(row[0]) if row else {}
