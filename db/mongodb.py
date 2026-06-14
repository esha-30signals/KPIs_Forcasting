import motor.motor_asyncio
import pymongo
import os
import re
from typing import Dict, Any, Optional, List, Tuple
from bson import ObjectId

# Global client and db instances
client = None
db = None
tls = True if os.getenv("MONGODB_TLS", "false") == "true" else False

# Get environment variables and strip whitespace
MONGODB_URI = os.getenv("MONGODB_URL", "").strip()
MONGO_DB_NAME = os.getenv("MONGODB_DB_NAME", "").strip()

if MONGODB_URI:
    # Mask password for logging
    sanitized_uri = re.sub(r'(://[^:]+:)([^@]+)(@)', r'\1****\3', MONGODB_URI)
else:
    sanitized_uri = "Not Set"

print(f"MONGODB_URI: {sanitized_uri}")
print(f"MONGO_DB_NAME: {MONGO_DB_NAME}")
print(f"tls: {tls}")

async def connect():
    """Initialize MongoDB connection (call once at startup)"""
    try:
        global client, db
        if client is None:
            if not MONGODB_URI:
                raise ValueError("MONGODB_URL environment variable is not set")
            if not MONGO_DB_NAME:
                raise ValueError("MONGODB_DB_NAME environment variable is not set")
            
            # Disable TLS/SSL certificate verification
            client = motor.motor_asyncio.AsyncIOMotorClient(
                MONGODB_URI,
                tls=tls,
                tlsInsecure=True,
                serverSelectionTimeoutMS=5000,
            )
            
            db = client[MONGO_DB_NAME]
            # Test the connection
            await client.admin.command("ping")
            print("✅ MongoDB connection established successfully")
        
        return db
    except Exception as e:
        if client:
            client.close()
            client = None
            db = None
        raise RuntimeError(f"❌ Database connection error: {str(e)}")


async def disconnect():
    """Close MongoDB connection (call at shutdown)"""
    global client, db
    if client:
        client.close()
        client = None
        db = None
        print("✅ MongoDB connection closed")


def get_db():
    """Get the database instance"""
    if db is None:
        raise RuntimeError("Database not initialized. Call connect() first.")
    return db


async def find_all(
    collection_name: str,
    query: Dict[str, Any] = None,
    sort: Optional[List[Tuple[str, int]]] = None,   # e.g., [("created_at", -1)]
    limit: Optional[int] = None,
    skip: Optional[int] = None,                     # offset
    projection: Optional[Dict[str, int]] = None     # e.g., {"_id": 0, "name": 1}
):
    try:
        # Set default query if None
        if query is None:
            query = {}
            
        database = get_db()
        collection = database[collection_name]

        # Validate sort parameter
        if sort is not None:
            if not isinstance(sort, list):
                raise ValueError("Sort must be a list of tuples")
            
            for item in sort:
                if not isinstance(item, tuple) or len(item) != 2:
                    raise ValueError("Sort items must be tuples of (field_name, direction)")
                
                field_name, direction = item
                if not isinstance(field_name, str):
                    raise ValueError("Sort field name must be a string")
                
                if direction not in [1, -1, pymongo.ASCENDING, pymongo.DESCENDING]:
                    raise ValueError("Sort direction must be 1/-1 or pymongo.ASCENDING/DESCENDING")

        cursor = collection.find(query, projection)

        # Apply sort first (this is important for performance)
        if sort:
            try:
                cursor = cursor.sort(sort)
                print(f"Applied sort: {sort}")  # Debug log
            except Exception as sort_error:
                print(f"Sort error: {sort_error}")  # Debug log
                raise ValueError(f"Sort operation failed: {str(sort_error)}")

        # Apply skip before limit
        if skip and skip > 0:
            cursor = cursor.skip(skip)

        if limit and limit > 0:
            cursor = cursor.limit(limit)

        results = []
        try:
            async for doc in cursor:
                doc = convert_objectid(doc)
                results.append(doc)
        except Exception as cursor_error:
            raise RuntimeError(f"Error processing cursor: {str(cursor_error)}")

        # print(f"Found {len(results)} documents")  # Debug log
        return results

    except Exception as e:
        print(f"find_all exception: {str(e)}")  # Debug log
        raise RuntimeError(f"find_all error: {str(e)}")


# Make this function async for consistency
async def find_one(collection_name: str, query: dict, projection: dict = None):
    try:
        database = get_db()
        collection = database[collection_name]
        result = await collection.find_one(query, projection)
        if result:
            result = convert_objectid(result)
        return result
    except Exception as e:
        raise RuntimeError(f"find_one error: {str(e)}")


async def insert_one(collection_name: str, document: dict):
    try:
        database = get_db()
        collection = database[collection_name]
        result = await collection.insert_one(document)
        return {"inserted_id": str(result.inserted_id)}
    except Exception as e:
        raise RuntimeError(f"insert_one error: {str(e)}")


async def insert_many(collection_name: str, documents: list):
    try:
        database = get_db()
        collection = database[collection_name]
        result = await collection.insert_many(documents)
        return {"inserted_ids": [str(_id) for _id in result.inserted_ids]}
    except Exception as e:
        print(e)
        raise RuntimeError(f"insert_many error: {str(e)}")


async def delete_many(collection_name: str, query: dict):
    try:
        database = get_db()
        collection = database[collection_name]
        result = await collection.delete_many(query)
        return {"deleted_count": result.deleted_count}
    except Exception as e:
        raise RuntimeError(f"delete_many error: {str(e)}")


async def update_one(collection_name: str, filter_query: dict, update_data: dict, upsert: bool = False):
    try:
        database = get_db()
        collection = database[collection_name]
        result = await collection.update_one(filter_query, update_data, upsert=upsert)
        return {
            "matched_count": result.matched_count,
            "modified_count": result.modified_count,
            "upserted_id": str(result.upserted_id) if result.upserted_id else None,
        }
    except Exception as e:
        raise RuntimeError(f"update_one error: {str(e)}")


async def update_many(collection_name: str, filter_query: dict, update_data: dict, upsert: bool = False):
    try:
        database = get_db()
        collection = database[collection_name]
        result = await collection.update_many(filter_query, update_data, upsert=upsert)
        return {
            "matched_count": result.matched_count,
            "modified_count": result.modified_count,
            "upserted_id": str(result.upserted_id) if result.upserted_id else None,
        }
    except Exception as e:
        raise RuntimeError(f"update_many error: {str(e)}")


async def aggregate(collection_name: str, pipeline: List[Dict[str, Any]]):
    """Execute an aggregation pipeline on a collection"""
    try:
        database = get_db()
        collection = database[collection_name]

        cursor = collection.aggregate(pipeline)
        results = []

        try:
            async for doc in cursor:
                doc = convert_objectid(doc)
                results.append(doc)
        except Exception as cursor_error:
            raise RuntimeError(f"Error processing aggregation cursor: {str(cursor_error)}")

        return results

    except Exception as e:
        raise RuntimeError(f"aggregate error: {str(e)}")


def convert_objectid(doc):
    """Recursively convert all ObjectId fields to str"""
    if isinstance(doc, dict):
        return {k: convert_objectid(v) for k, v in doc.items()}
    elif isinstance(doc, list):
        return [convert_objectid(item) for item in doc]
    elif isinstance(doc, ObjectId):
        return str(doc)
    else:
        return doc


# Helper function to test sort functionality
async def test_sort(collection_name: str, sort_field: str):
    """Test function to verify sort is working"""
    try:
        # Test both ascending and descending
        asc_results = await find_all(collection_name, sort=[(sort_field, 1)], limit=5)
        desc_results = await find_all(collection_name, sort=[(sort_field, -1)], limit=5)
        
        print(f"Ascending sort on {sort_field}:")
        for doc in asc_results:
            print(f"  {doc.get(sort_field, 'N/A')}")
            
        print(f"Descending sort on {sort_field}:")
        for doc in desc_results:
            print(f"  {doc.get(sort_field, 'N/A')}")
            
        return {"ascending": asc_results, "descending": desc_results}
    except Exception as e:
        print(f"Sort test error: {e}")
        return None

async def count(collection_name: str, query: dict = None):
    try:
        database = get_db()
        collection = database[collection_name]
        result = await collection.count_documents(query)
        return result
    except Exception as e:
        raise RuntimeError(f"count error: {str(e)}")