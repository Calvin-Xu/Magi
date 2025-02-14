# src/magi/main.py
import sys
import requests
from gqlalchemy import Memgraph
import psycopg2
from pyspark.sql import SparkSession

def check_memgraph(host="memgraph", port=7687):
    try:
        mg = Memgraph(host=host, port=port)
        mg.execute("RETURN 1;")
        print("[OK] Connected to Memgraph at", f"{host}:{port}")
    except Exception as e:
        print("[ERROR] Could not connect to Memgraph:", e)
        sys.exit(1)

def check_postgres(host="postgres", port=5432, db="magidb", user="magiuser", password="magipassword"):
    try:
        conn = psycopg2.connect(
            host=host,
            port=port,
            dbname=db,
            user=user,
            password=password
        )
        print("[OK] Connected to Postgres at", f"{host}:{port}")
        conn.close()
    except Exception as e:
        print("[ERROR] Could not connect to Postgres:", e)
        sys.exit(1)

def check_spark(master_url="spark://spark:7077"):
    try:
        spark = SparkSession.builder \
            .master(master_url) \
            .appName("MagiHealthCheck") \
            .getOrCreate()
        print("[OK] Spark session created successfully.")
        spark.stop()
    except Exception as e:
        print("[ERROR] Could not create Spark session:", e)
        sys.exit(1)

def check_memgraph_lab(host="memgraph-lab", port=3000):
    try:
        url = f"http://{host}:{port}"
        resp = requests.get(url, timeout=3)
        if resp.status_code == 200:
            print("[OK] Memgraph Lab is responding at", url)
        else:
            print("[ERROR] Memgraph Lab not OK. HTTP status:", resp.status_code)
            sys.exit(1)
    except Exception as e:
        print("[ERROR] Could not connect to Memgraph Lab:", e)
        sys.exit(1)

def main():
    print("Starting Magi health checks...")

    # Check if Memgraph is running
    check_memgraph()

    # Check if Postgres is running
    check_postgres()

    # Check if Spark is running
    check_spark()

    # Check if Memgraph Lab is accessible
    check_memgraph_lab()

    print("All services appear to be running!")

if __name__ == "__main__":
    main()
