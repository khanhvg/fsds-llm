import logging
import psycopg
from typing import List, Dict
from psycopg.types.json import Jsonb
from datetime import datetime, timedelta
import random
import uuid
import json

logger = logging.getLogger(__name__)

# Database connection
DB_CONNECTION = "postgresql://postgres:postgres@localhost:5432/postgres"

# Gundam data for generating products
GUNDAM_SERIES = [
    "Mobile Suit Gundam",
    "Mobile Suit Zeta Gundam",
    "Mobile Suit Gundam ZZ",
    "Mobile Suit Gundam: Char's Counterattack",
    "Mobile Suit Gundam F91",
    "Mobile Suit Victory Gundam",
    "Mobile Fighter G Gundam",
    "Mobile Suit Gundam Wing",
    "After War Gundam X",
    "Turn A Gundam",
    "Mobile Suit Gundam SEED",
    "Mobile Suit Gundam 00",
    "Mobile Suit Gundam Unicorn",
    "Mobile Suit Gundam Iron-Blooded Orphans",
    "Mobile Suit Gundam: The Witch from Mercury",
]

GUNDAM_GRADES = [
    "High Grade (HG)",
    "Real Grade (RG)",
    "Master Grade (MG)",
    "Perfect Grade (PG)",
    "Super Deformed (SD)",
    "Entry Grade (EG)",
    "No Grade (NG)",
    "RE/100",
    "Full Mechanics",
]

GUNDAM_SCALES = ["1/144", "1/100", "1/60", "Non-scale"]

GUNDAM_MODELS = [
    # Mobile Suit Gundam
    {"name": "RX-78-2 Gundam", "series": "Mobile Suit Gundam"},
    {"name": "MS-06S Zaku II Commander Type", "series": "Mobile Suit Gundam"},
    {"name": "MS-07B Gouf", "series": "Mobile Suit Gundam"},
    {"name": "RX-77-2 Guncannon", "series": "Mobile Suit Gundam"},
    {"name": "RX-75 Guntank", "series": "Mobile Suit Gundam"},
    {"name": "MS-09 Dom", "series": "Mobile Suit Gundam"},
    {"name": "MSM-07 Z'Gok", "series": "Mobile Suit Gundam"},
    # Zeta Gundam
    {"name": "MSZ-006 Zeta Gundam", "series": "Mobile Suit Zeta Gundam"},
    {"name": "RX-178 Gundam Mk-II", "series": "Mobile Suit Zeta Gundam"},
    {"name": "MSN-00100 Hyaku Shiki", "series": "Mobile Suit Zeta Gundam"},
    {"name": "PMX-003 The O", "series": "Mobile Suit Zeta Gundam"},
    # Gundam ZZ
    {"name": "MSZ-010 ZZ Gundam", "series": "Mobile Suit Gundam ZZ"},
    {"name": "AMX-004 Qubeley", "series": "Mobile Suit Gundam ZZ"},
    # Char's Counterattack
    {"name": "RX-93 Nu Gundam", "series": "Mobile Suit Gundam: Char's Counterattack"},
    {"name": "MSN-04 Sazabi", "series": "Mobile Suit Gundam: Char's Counterattack"},
    # Gundam F91
    {"name": "F91 Gundam F91", "series": "Mobile Suit Gundam F91"},
    # Victory Gundam
    {"name": "LM314V21 Victory Gundam", "series": "Mobile Suit Victory Gundam"},
    # G Gundam
    {"name": "GF13-017NJ Shining Gundam", "series": "Mobile Fighter G Gundam"},
    {"name": "GF13-017NJII God Gundam", "series": "Mobile Fighter G Gundam"},
    {"name": "GF13-001NHII Master Gundam", "series": "Mobile Fighter G Gundam"},
    # Gundam Wing
    {"name": "XXXG-01W Wing Gundam", "series": "Mobile Suit Gundam Wing"},
    {"name": "XXXG-00W0 Wing Gundam Zero", "series": "Mobile Suit Gundam Wing"},
    {"name": "XXXG-01D Gundam Deathscythe", "series": "Mobile Suit Gundam Wing"},
    {"name": "XXXG-01H Gundam Heavyarms", "series": "Mobile Suit Gundam Wing"},
    {"name": "XXXG-01S Gundam Sandrock", "series": "Mobile Suit Gundam Wing"},
    {"name": "XXXG-01N Gundam Nataku", "series": "Mobile Suit Gundam Wing"},
    {"name": "OZ-13MS Gundam Epyon", "series": "Mobile Suit Gundam Wing"},
    # Gundam X
    {"name": "GX-9900 Gundam X", "series": "After War Gundam X"},
    # Turn A Gundam
    {"name": "∀ Gundam", "series": "Turn A Gundam"},
    # Gundam SEED
    {"name": "GAT-X105 Strike Gundam", "series": "Mobile Suit Gundam SEED"},
    {"name": "ZGMF-X10A Freedom Gundam", "series": "Mobile Suit Gundam SEED"},
    {"name": "ZGMF-X09A Justice Gundam", "series": "Mobile Suit Gundam SEED"},
    {"name": "GAT-X303 Aegis Gundam", "series": "Mobile Suit Gundam SEED"},
    {"name": "GAT-X207 Blitz Gundam", "series": "Mobile Suit Gundam SEED"},
    {"name": "GAT-X102 Duel Gundam", "series": "Mobile Suit Gundam SEED"},
    {"name": "GAT-X103 Buster Gundam", "series": "Mobile Suit Gundam SEED"},
    {
        "name": "ZGMF-X20A Strike Freedom Gundam",
        "series": "Mobile Suit Gundam SEED Destiny",
    },
    {
        "name": "ZGMF-X19A Infinite Justice Gundam",
        "series": "Mobile Suit Gundam SEED Destiny",
    },
    # Gundam 00
    {"name": "GN-001 Gundam Exia", "series": "Mobile Suit Gundam 00"},
    {"name": "GN-002 Gundam Dynames", "series": "Mobile Suit Gundam 00"},
    {"name": "GN-003 Gundam Kyrios", "series": "Mobile Suit Gundam 00"},
    {"name": "GN-004 Gundam Nadleeh", "series": "Mobile Suit Gundam 00"},
    {"name": "GN-005 Gundam Virtue", "series": "Mobile Suit Gundam 00"},
    {"name": "GN-0000 00 Gundam", "series": "Mobile Suit Gundam 00"},
    {"name": "GN-0000+GNR-010 00 Raiser", "series": "Mobile Suit Gundam 00"},
    # Gundam Unicorn
    {"name": "RX-0 Unicorn Gundam", "series": "Mobile Suit Gundam Unicorn"},
    {"name": "RX-0 Unicorn Gundam 02 Banshee", "series": "Mobile Suit Gundam Unicorn"},
    {"name": "RX-0 Unicorn Gundam 03 Phenex", "series": "Mobile Suit Gundam Unicorn"},
    {"name": "MSN-06S Sinanju", "series": "Mobile Suit Gundam Unicorn"},
    # Gundam Iron-Blooded Orphans
    {
        "name": "ASW-G-08 Gundam Barbatos",
        "series": "Mobile Suit Gundam Iron-Blooded Orphans",
    },
    {
        "name": "ASW-G-08 Gundam Barbatos Lupus",
        "series": "Mobile Suit Gundam Iron-Blooded Orphans",
    },
    {
        "name": "ASW-G-08 Gundam Barbatos Lupus Rex",
        "series": "Mobile Suit Gundam Iron-Blooded Orphans",
    },
    {
        "name": "ASW-G-11 Gundam Gusion Rebake",
        "series": "Mobile Suit Gundam Iron-Blooded Orphans",
    },
    # Gundam: The Witch from Mercury
    {
        "name": "XVX-016 Gundam Aerial",
        "series": "Mobile Suit Gundam: The Witch from Mercury",
    },
    {
        "name": "XVX-016RN Gundam Aerial Rebuild",
        "series": "Mobile Suit Gundam: The Witch from Mercury",
    },
    {
        "name": "XGF-01 Gundam Calibarn",
        "series": "Mobile Suit Gundam: The Witch from Mercury",
    },
    {
        "name": "XGF-02 Gundam Lachesis",
        "series": "Mobile Suit Gundam: The Witch from Mercury",
    },
]


def generate_description(name, grade, scale, series):
    """Generate a realistic description for a Gundam model kit"""
    descriptions = [
        f"The {name} {grade} {scale} scale model kit from {series}. Features excellent articulation and stunning detail.",
        f"Build the iconic {name} from {series} with this {grade} {scale} scale model kit. Includes multiple weapons and accessories.",
        f"This {grade} {scale} {name} model kit showcases the memorable design from {series}. Perfect for collectors and builders alike.",
        f"{grade} {scale} scale model kit of the {name} from {series}. Features detailed panel lines and great posability.",
        f"Recreate the famous scenes from {series} with this {grade} {name} model kit in {scale} scale. Includes display stand.",
    ]
    return random.choice(descriptions)


def generate_products(count=100):
    """Generate Gundam product data"""
    products = []

    # Ensure we have at least count products by generating combinations
    for i in range(count):
        if i < len(GUNDAM_MODELS):
            # Use predefined models first
            model = GUNDAM_MODELS[i].copy()
            series = model["series"]
        else:
            # For additional products, randomly select from models again
            model = random.choice(GUNDAM_MODELS).copy()
            series = model["series"]

        # Randomize other attributes
        grade = random.choice(GUNDAM_GRADES)
        scale = random.choice(GUNDAM_SCALES)

        # Set appropriate category based on grade
        if "High Grade" in grade:
            category = "HG"
        elif "Real Grade" in grade:
            category = "RG"
        elif "Master Grade" in grade:
            category = "MG"
        elif "Perfect Grade" in grade:
            category = "PG"
        elif "Super Deformed" in grade:
            category = "SD"
        else:
            category = "Other"

        # Generate price based on grade
        if category == "PG":
            price = round(random.uniform(180.0, 300.0), 2)
        elif category == "MG":
            price = round(random.uniform(50.0, 100.0), 2)
        elif category == "RG":
            price = round(random.uniform(30.0, 60.0), 2)
        elif category == "HG":
            price = round(random.uniform(15.0, 40.0), 2)
        elif category == "SD":
            price = round(random.uniform(10.0, 25.0), 2)
        else:
            price = round(random.uniform(20.0, 80.0), 2)

        # Generate random release date within the last 10 years
        days_back = random.randint(0, 365 * 10)
        release_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")

        # Generate random stock quantity
        stock = random.randint(0, 50)

        # Generate placeholder image URL
        image_url = f"https://example.com/gundam/images/{model['name'].replace(' ', '_').lower()}.jpg"

        # Generate description
        description = generate_description(model["name"], grade, scale, series)

        # Create product object
        product = {
            "name": f"{model['name']} {grade} {scale}",
            "description": description,
                "price": price,
            "category": category,
            "series": series,
            "scale": scale,
            "grade": grade,
            "release_date": release_date,
            "stock_quantity": stock,
            "image_url": image_url,
        }

        products.append(product)

    return products


def insert_products(products):
    """Insert products into database"""
    try:
        with psycopg.connect(DB_CONNECTION) as conn:
            with conn.cursor() as cur:
                for product in products:
                cur.execute(
                    """
                        INSERT INTO products 
                        (name, description, price, category, series, scale, grade, release_date, stock_quantity, image_url)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            product["name"],
                            product["description"],
                            product["price"],
                            product["category"],
                            product["series"],
                            product["scale"],
                            product["grade"],
                            product["release_date"],
                            product["stock_quantity"],
                            product["image_url"],
                        ),
                    )
            conn.commit()
        print(f"Successfully inserted {len(products)} products")
        except Exception as e:
        print(f"Error inserting products: {e}")


def main():
    # Generate products
    products = generate_products(100)

    # Save products to JSON file for reference
    with open("gundam_products.json", "w") as f:
        json.dump(products, f, indent=2)
    print(f"Saved product data to gundam_products.json")

    # Insert products into database
    insert_products(products)


if __name__ == "__main__":
    main()
