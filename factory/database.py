from datetime import datetime
from bson import ObjectId
import firebase_admin
from firebase_admin import credentials,firestore

class Database(object):
    def __init__(self):
        credentialData = credentials.Certificate("factory/credentials.json")
        firebase_admin.initialize_app(credentialData)
        self.db = firestore.client()

    def insert(self, element):
        element["created"] = datetime.now()
        element["updated"] = datetime.now()

        self.db.collection(u'nlpdb').add(element)
        return 'Başarılı'

    def get(self):
        return self.db.collection(u'nlpdb').get()

    def find(self, criteria, collection_name, projection=None, sort=None, limit=0, cursor=False):

        if "_id" in criteria:
            criteria["_id"] = ObjectId(criteria["_id"])

        found = self.db[collection_name].find(filter=criteria, projection=projection, limit=limit, sort=sort)

        if cursor:
            return found

        found = list(found)

        for i in range(len(found)):
            if "_id" in found[i]:
                found[i]["_id"] = str(found[i]["_id"])

        return found

    def find_by_id(self, id, collection_name):
        found = self.db[collection_name].find_one({"_id": ObjectId(id)})
        
        if found is None:
            return not found
        
        if "_id" in found:
             found["_id"] = str(found["_id"])

        return found

    def update(self, id, element, collection_name):
        criteria = {"_id": ObjectId(id)}

        element["updated"] = datetime.now()
        set_obj = {"$set": element}
        updated = self.db[collection_name].update_one(criteria, set_obj)
        if updated.matched_count == 1:
            return "Successfully Updated"

    def delete(self, id, collection_name):
        deleted = self.db[collection_name].delete_one({"_id": ObjectId(id)})
        return bool(deleted.deleted_count)
