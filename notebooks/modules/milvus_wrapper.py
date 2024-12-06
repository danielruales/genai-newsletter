from pymilvus import (
    MilvusClient,  # Milvus lite
    connections, utility, FieldSchema, CollectionSchema, Collection, DataType,
    AnnSearchRequest,
    WeightedRanker,
)

# Import the BGEM3EmbeddingFunction which is used to create dense vectors
from milvus_model.hybrid import BGEM3EmbeddingFunction

class MilvusLiteClient:
    def __init__(self, uri="./milvus_demo.db"):
        # Initialize Milvus Lite client
        self.client = MilvusClient(uri=uri)

    def create_collection(self, collection_name, dimension, schema=None, index_params=None, metric_type=None, consistency_level="Strong", drop_if_exists=False):
        if self.client.has_collection(collection_name):
            if not drop_if_exists:  # Raise an error if the collection already exists and drop_if_exists is False
                raise ValueError("Collection already exists")
            else:
                self.client.drop_collection(collection_name)
        self.client.create_collection(
            collection_name=collection_name,
            dimension=dimension,
            schema=schema,
            index_params=index_params,
            consistency_level=consistency_level,
            metric_type=metric_type
        )

    def create_demo_hybrid_schema(self, embedding_dim):
        schema = self.client.create_schema(
            auto_id=True,
            enable_dynamic_field=True,
            description="Collection for hybrid search with vector and text fields"
        )
        schema.add_field(
            field_name="id",
            datatype=DataType.INT64,
            is_primary=True
        )
        schema.add_field(
            field_name="dense_vector",
            datatype=DataType.FLOAT_VECTOR,
            dim=embedding_dim
        )
        schema.add_field(
            field_name="sparse_vector",
            datatype=DataType.SPARSE_FLOAT_VECTOR,
        )
        schema.add_field(
            field_name="text",
            datatype=DataType.VARCHAR,
            max_length=500
        )
        return schema

    def create_demo_hybrid_indices(self):
        index_params = self.client.prepare_index_params()
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

class MilvusFullClient:
    def __init__(self, uri="./milvus_demo.db"):
        # Initialize full Milvus client
        connections.connect(uri=uri)

    def create_collection(self, collection_name, schema, index_params=None, metric_type=None, consistency_level="Strong", drop_if_exists=False):
        if utility.has_collection(collection_name):
            if not drop_if_exists:  # Raise an error if the collection already exists and drop_if_exists is False
                raise ValueError("Collection already exists")
            else:
                Collection(collection_name).drop()
        col = Collection(collection_name, schema, consistency_level="Strong")
        return col

    def create_demo_hybrid_schema(self, embedding_dim):
        fields = [
            FieldSchema(
                name="id", dtype=DataType.VARCHAR, is_primary=True, auto_id=True, max_length=100
            ),
            FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=embedding_dim),
            FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=20000),
        ]
        schema = CollectionSchema(fields)
        return schema

    def create_demo_hybrid_indices(self, col):
        sparse_index = {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"}
        col.create_index("sparse_vector", sparse_index)
        dense_index = {"index_type": "AUTOINDEX", "metric_type": "IP"}
        col.create_index("dense_vector", dense_index)
        col.load()
    
    def dense_search(self, col, query_dense_embedding, limit=10):
        search_params = {"metric_type": "IP", "params": {}}
        res = col.search(
            [query_dense_embedding],
            anns_field="dense_vector",
            limit=limit,
            output_fields=["text"],
            param=search_params,
        )[0]
        return [hit.get("text") for hit in res]

    def sparse_search(self, col, query_sparse_embedding, limit=10):
        search_params = {
            "metric_type": "IP",
            "params": {},
        }
        res = col.search(
            [query_sparse_embedding],
            anns_field="sparse_vector",
            limit=limit,
            output_fields=["text"],
            param=search_params,
        )[0]
        return [hit.get("text") for hit in res]

    
    def hybrid_search(
        self,
        col,
        query_dense_embedding,
        query_sparse_embedding,
        sparse_weight=1.0,
        dense_weight=1.0,
        limit=10,
    ):
        dense_search_params = {"metric_type": "IP", "params": {}}
        dense_req = AnnSearchRequest(
            [query_dense_embedding], "dense_vector", dense_search_params, limit=limit
        )
        sparse_search_params = {"metric_type": "IP", "params": {}}
        sparse_req = AnnSearchRequest(
            [query_sparse_embedding], "sparse_vector", sparse_search_params, limit=limit
        )
        rerank = WeightedRanker(sparse_weight, dense_weight)
        res = col.hybrid_search(
            [sparse_req, dense_req], rerank=rerank, limit=limit, output_fields=["text"]
        )[0]
        return [hit.get("text") for hit in res]



def get_dense_embedding_details(use_fp16=False, device="cpu"):
    dense_embedding_function = BGEM3EmbeddingFunction(use_fp16=use_fp16, device=device)
    dense_dim = dense_embedding_function.dim["dense"]
    return dense_dim, dense_embedding_function