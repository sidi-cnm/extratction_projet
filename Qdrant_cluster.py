from qdrant_client import QdrantClient

qdrant_client = QdrantClient(
    url="https://bf703566-9c4d-4187-9d56-8b3f5bd05dbb.europe-west3-0.gcp.cloud.qdrant.io:6333", 
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.5HIJO_6SMFXnoTydC8Xcl2PgCFz3TCTTdBPwWKltRWY",
)

points = qdrant_client.scroll(
    collection_name="medical_records",
    limit=5  # nombre de points à récupérer
)

for point in points:
    print("ID:", point.id)
    print("Payload:", point.payload)
    print("Vector:", point.vector[:10], "...")  # 