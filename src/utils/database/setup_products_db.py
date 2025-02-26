import os
import psycopg
import subprocess

# Database connection
DB_CONNECTION = "postgresql://postgres:postgres@localhost:5432/postgres"


def run_sql_file(file_path):
    """Run SQL file against the database"""
    try:
        # Read SQL file
        with open(file_path, "r") as f:
            sql = f.read()

        # Execute SQL
        with psycopg.connect(DB_CONNECTION) as conn:
            with conn.cursor() as cur:
                cur.execute(sql)
            conn.commit()
        print(f"Successfully executed {file_path}")
        return True
    except Exception as e:
        print(f"Error executing {file_path}: {e}")
        return False


def main():
    # Get current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Execute SQL schema file
    schema_path = os.path.join(current_dir, "create_products.sql")
    if not run_sql_file(schema_path):
        print("Failed to create schema. Aborting.")
        return

    # Run data insertion script
    print("Creating products data...")
    insert_script = os.path.join(current_dir, "products_insert.py")
    subprocess.run(["python", insert_script])

    print("Product database setup complete!")


if __name__ == "__main__":
    main()
