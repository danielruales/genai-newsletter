from pymilvus import (
    MilvusClient, # Milvus lite
    connections, utility # Milvus connection
    ,FieldSchema, CollectionSchema, Collection, DataType
)

# Import the BGEM3EmbeddingFunction which is used to create dense vectors
# This is used instead of the openai embedding function because it is faster
from milvus_model.hybrid import BGEM3EmbeddingFunction


def get_milvus_client(uri="./milvus_demo.db", milvus_lite=True):
    # Can set uri to the local path for the Milvus database
    if milvus_lite:
        milvus_client = MilvusClient(uri=uri)
        return milvus_client
    else:
        connections.connect(uri=uri)

def create_milvus_collection(collection_name, milvus_client=None, dimension=None, schema=None, index_params=None, metric_type=None, consistency_level="Strong",  drop_if_exists=False):

    if milvus_client:
        if milvus_client.has_collection(collection_name):
            if drop_if_exists == False: # Raise an error if the collection already exists and drop_if_exists is False
                raise ValueError("Collection already exists")
            else:
                milvus_client.drop_collection(collection_name)
        milvus_client.create_collection(
            collection_name=collection_name,
            dimension=dimension,
            schema=schema,
            index_params=index_params,
            consistency_level=consistency_level,
            metric_type=metric_type
        )
        return None
    else:
        # Create collection (drop the old one if exists)
        if utility.has_collection(collection_name):
            if drop_if_exists == False: # Raise an error if the collection already exists and drop_if_exists is False
                raise ValueError("Collection already exists")
            else:
                Collection(collection_name).drop()
        col = Collection(collection_name, schema, consistency_level="Strong")
        return col

def create_demo_hybrid_milvus_schema(embedding_dim, milvus_client=None):
    if milvus_client:
        schema = milvus_client.create_schema(
            auto_id=True,
            enable_dynamic_field=True,
            description="Collection for hybrid search with vector and text fields"
        )

        # Use auto generated id as primary key
        schema.add_field(
            field_name="id",
            datatype=DataType.INT64,
            is_primary=True
        )

        # Milvus now supports both sparse and dense vectors,
        # We can store each in a separate field to conduct hybrid search on both vectors
        schema.add_field(
            field_name="dense_vector",
            datatype=DataType.FLOAT_VECTOR,
            dim=embedding_dim
        )

        schema.add_field(
            field_name="sparse_vector",
            datatype=DataType.SPARSE_FLOAT_VECTOR,
            # dim=embedding_dim
        )

        # Store the original text to retrieve based on semantically distance
        schema.add_field(
            field_name="text",
            datatype=DataType.VARCHAR,
            max_length=500
        )
    else:
        # Specify the data schema for the new Collection
        fields = [
            # Use auto generated id as primary key
            FieldSchema(
                name="id", dtype=DataType.VARCHAR, is_primary=True, auto_id=True, max_length=100
            ),
            
            # Store each dense and sparse vectors in separate fields to conduct hybrid search on both vectors
            FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=embedding_dim),
            FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),

            # Store the original text to retrieve based on semantically distance
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=512),
        ]
        schema = CollectionSchema(fields)
    return schema

def get_dense_embedding_details(use_fp16=False, device="cpu"):
    dense_embedding_function = BGEM3EmbeddingFunction(use_fp16=False, device="cpu")
    dense_dim = dense_embedding_function.dim["dense"] 
    return dense_dim, dense_embedding_function

def create_demo_hybrid_milvus_indices(milvus_client=None, milvus_collection=None):
    if milvus_client:
        index_params = milvus_client.prepare_index_params()
        index_params.add_index(
            field_name="sparse_vector",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="IP",
        )

        index_params.add_index(
            field_name="dense_vector",
            index_type="AUTOINDEX",
            metric_type="IP",
        )

        return index_params
    else:
        # To make vector search efficient, we need to create indices for the vector fields
        sparse_index = {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"}
        milvus_collection.create_index("sparse_vector", sparse_index)
        dense_index = {"index_type": "AUTOINDEX", "metric_type": "IP"}
        milvus_collection.create_index("dense_vector", dense_index)
        milvus_collection.load()

