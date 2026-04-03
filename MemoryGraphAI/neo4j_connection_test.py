# from neo4j import GraphDatabase

# uri = "neo4j+s://170d5454.databases.neo4j.io"
# user = "neo4j"
# password = "Zsy0STAu9DAN7zzWYeUETmF1VwD-hFUK5utP4O_OjNU"

# driver = GraphDatabase.driver(uri, auth=(user, password))

# try:
#     with driver.session() as session:
#         result = session.run("RETURN 1")
#         print("✅ Connected:", result.single()[0])
# except Exception as e:
#     print("❌ Error:", e)

# driver.close()

from neo4j import GraphDatabase

driver = GraphDatabase.driver(
    "bolt://170d5454.databases.neo4j.io:7687",
    auth=("neo4j", "Zsy0STAu9DAN7zzWYeUETmF1VwD-hFUK5utP4O_OjNU"),
    encrypted=False    
)

with driver.session() as session:
    print(session.run("RETURN 1").single()[0])

driver.close()