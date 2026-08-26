import pandas as pd
import psycopg2
import os

#==========================================================
# CONFIG: PATHS
#==========================================================

BASE = "/nfs/home/users/vpandya/integration_to_KB"

PATH_H_LATENT = f"{BASE}/h_latent_full.csv"
PATH_TILE_REG = f"{BASE}/tile_reg_full.csv"
PATH_SURVIVAL = f"{BASE}/survival.csv"

#==========================================================
# STEP 1: LOAD CSV FILES
#==========================================================

def load_csv(path):
    print(f"Loading {path}")
    return pd.read_csv(path)



df_z = pd.DataFrame(z_latent)
df_z.columns = [f"z_{i}" for i in range(df_z.shape[1])]

df_z["tile_name"] = tiles
df_z["slide_id"] = slides
df_z["sample_id"] = samples

df_z.to_csv("z_latent_full.csv", index=False)
print("Saved z_latent_full.csv")


from tensorflow import keras
model = keras.models.load_model("/nfs/home/users/vpandya/hierarchical_classifier_best.keras")


X = df_z[[c for c in df_z.columns if c.startswith("z_")]].values
preds = model.predict(X)

#==========================================================
# STEP 2: COPY FUNCTION
#==========================================================

def copy_to_pg(conn, df, table):
    cursor = conn.cursor()
    print(f"Loading → {table} ({len(df)} rows)")

    tmp_path = f"/tmp/{table}.csv"
    df.to_csv(tmp_path, index=False)

    # Columns named explicitly. Without a column list, COPY matches the CSV
    # positionally against *every* column in the table, so adding one to
    # tile_registry (the assignment-confidence columns) would break this load
    # even though the CSV never changed — and the error points at a type
    # mismatch on some unrelated column rather than at the real cause.
    columns = ", ".join(f'"{c}"' for c in df.columns)

    with open(tmp_path, "r") as f:
        cursor.copy_expert(f"COPY {table} ({columns}) FROM STDIN CSV HEADER", f)

    conn.commit()
    cursor.close()







#==========================================================
# STEP 3: MAIN PIPELINE
#==========================================================

def run():
    print("\n=== Step 1: Load files ===")
    h_latent_df = load_csv(PATH_H_LATENT)
    tile_df = load_csv(PATH_TILE_REG)
    survival_df = load_csv(PATH_SURVIVAL)

    print("\n=== Step 2: Connect to Postgres ===")
    conn = psycopg2.connect(
        dbname="hpl_kb",
        user="vpandya",
        password="",
        host="localhost",
        port="5432"
    )

    print("\n=== Step 3: Truncate only needed tables ===")
    truncate_query = """
    TRUNCATE h_latent_vectors,
             tile_registry,
             hpc_survival_analysis
    RESTART IDENTITY;
    """
    cur = conn.cursor()
    cur.execute(truncate_query)
    conn.commit()

    print("\n=== Step 4: Load tables ===")

    copy_to_pg(conn, h_latent_df, "h_latent_vectors")
    copy_to_pg(conn, tile_df, "tile_registry")
    copy_to_pg(conn, survival_df, "hpc_survival_analysis")

    print("\n=== DONE: KB Updated Successfully ===")

#==========================================================
# ENTRY
#==========================================================

if __name__ == "__main__":
    run()