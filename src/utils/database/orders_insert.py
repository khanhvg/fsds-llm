import logging
import psycopg
from typing import List, Dict
from psycopg.types.json import Jsonb
from datetime import datetime, timedelta
import random
import uuid

logger = logging.getLogger(__name__)

# Add ORDER_STATUS to the constants
ORDER_STATUS = ["pending", "in_production", "shipped"]

# List of Gundam items
GUNDAM_ITEMS = [
    {
        "model": "RX-78-2 Gundam",
        "grade": "Perfect Grade",
        "base_price": 180.00,
        "scale": "1/60",
    },
    {
        "model": "Unicorn Gundam",
        "grade": "Real Grade",
        "base_price": 45.00,
        "scale": "1/144",
    },
    {
        "model": "Strike Freedom Gundam",
        "grade": "Master Grade",
        "base_price": 65.00,
        "scale": "1/100",
    },
    {
        "model": "Gundam Barbatos",
        "grade": "Master Grade",
        "base_price": 55.00,
        "scale": "1/100",
    },
    {"model": "Sazabi", "grade": "Real Grade", "base_price": 50.00, "scale": "1/144"},
    {
        "model": "Nu Gundam",
        "grade": "Real Grade",
        "base_price": 48.00,
        "scale": "1/144",
    },
    {
        "model": "Wing Gundam Zero EW",
        "grade": "Perfect Grade",
        "base_price": 190.00,
        "scale": "1/60",
    },
    {"model": "Zaku II", "grade": "High Grade", "base_price": 25.00, "scale": "1/144"},
    {
        "model": "Gundam Exia",
        "grade": "Perfect Grade",
        "base_price": 175.00,
        "scale": "1/60",
    },
    {
        "model": "Sinanju",
        "grade": "Master Grade",
        "base_price": 70.00,
        "scale": "1/100",
    },
]

# List of customers
CUSTOMERS = [
    {"name": "John Smith", "email": "john.smith@email.com"},
    {"name": "Emma Wilson", "email": "emma.w@email.com"},
    {"name": "Michael Chen", "email": "m.chen@email.com"},
    {"name": "Sarah Johnson", "email": "sarahj@email.com"},
    {"name": "David Kim", "email": "d.kim@email.com"},
    {"name": "Lisa Garcia", "email": "l.garcia@email.com"},
    {"name": "James Williams", "email": "jwilliams@email.com"},
    {"name": "Maria Rodriguez", "email": "m.rodriguez@email.com"},
    {"name": "Robert Taylor", "email": "rob.t@email.com"},
    {"name": "Jennifer Lee", "email": "j.lee@email.com"},
    {"name": "William Brown", "email": "w.brown@email.com"},
    {"name": "Elizabeth Davis", "email": "e.davis@email.com"},
    {"name": "Thomas Anderson", "email": "t.anderson@email.com"},
    {"name": "Jessica Martinez", "email": "j.martinez@email.com"},
    {"name": "Daniel White", "email": "d.white@email.com"},
    {"name": "Michelle Turner", "email": "m.turner@email.com"},
    {"name": "Kevin Parker", "email": "k.parker@email.com"},
    {"name": "Amanda Collins", "email": "a.collins@email.com"},
    {"name": "Christopher Hall", "email": "c.hall@email.com"},
    {"name": "Rachel Green", "email": "r.green@email.com"},
]


class GundamOrder:
    def __init__(
        self,
        order_id: str,
        order_detail: Dict,
        total_price: float,
        customer_name: str,
        customer_email: str,
        status: str = "pending",
        created_at: datetime = None,
        updated_at: datetime = None,
    ):
        self.order_id = order_id
        self.order_detail = order_detail
        self.total_price = total_price
        self.customer_name = customer_name
        self.customer_email = customer_email
        self.status = status
        self.created_at = created_at or datetime.now()
        self.updated_at = updated_at or datetime.now()

    def to_tuple(self):
        """Convert order to tuple for database insertion"""
        return (
            self.order_id,
            Jsonb(self.order_detail),
            self.total_price,
            self.customer_name,
            self.customer_email,
            self.status,
            self.created_at,
            self.updated_at,
        )


def generate_random_order() -> GundamOrder:
    """Generate a random order with 1-3 items"""
    # Select random customer
    customer = random.choice(CUSTOMERS)

    # Select random status
    status = random.choice(ORDER_STATUS)

    # Generate random number of items (1-3)
    num_items = random.randint(1, 3)

    # Select random items
    selected_items = random.sample(GUNDAM_ITEMS, num_items)

    # Generate order details
    items = []
    total_price = 0

    for item in selected_items:
        quantity = random.randint(1, 2)
        price = item["base_price"]
        items.append(
            {
                "model": item["model"],
                "grade": item["grade"],
                "scale": item["scale"],
                "quantity": quantity,
                "price": price,
            }
        )
        total_price += price * quantity

    order_detail = {"items": items}

    # Create timestamps based on status
    current_time = datetime.now()
    if status == "pending":
        created_at = current_time
        updated_at = current_time
    elif status == "in_production":
        # Set created_at to 1-3 days ago
        created_at = current_time - timedelta(days=random.randint(1, 3))
        updated_at = current_time
    else:  # shipped
        # Set created_at to 4-7 days ago
        created_at = current_time - timedelta(days=random.randint(4, 7))
        updated_at = current_time

    return GundamOrder(
        order_id=f"ORD-{uuid.uuid4().hex[:8].upper()}",
        order_detail=order_detail,
        total_price=round(total_price, 2),
        customer_name=customer["name"],
        customer_email=customer["email"],
        status=status,
        created_at=created_at,
        updated_at=updated_at,
    )


def generate_random_orders(num_orders: int) -> List[GundamOrder]:
    """Generate a specified number of random orders"""
    return [generate_random_order() for _ in range(num_orders)]


class PGOrders:
    def __init__(self):
        self._init_db()

    def _init_db(self):
        with psycopg.connect(
            "postgresql://postgres:postgres@localhost:5432/postgres"
        ) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
          CREATE TABLE IF NOT EXISTS orders (
          order_id VARCHAR(50) PRIMARY KEY,
          order_detail JSONB NOT NULL,
          total_price DECIMAL(10,2) NOT NULL,
          customer_name VARCHAR(100) NOT NULL,
          customer_email VARCHAR(100) NOT NULL,
          status VARCHAR(20) NOT NULL,
          created_at TIMESTAMP NOT NULL,
          updated_at TIMESTAMP NOT NULL);
          CREATE INDEX idx_orders_email ON orders(customer_email);
        CREATE INDEX idx_orders_status ON orders(status);
        CREATE INDEX idx_orders_created_at ON orders(created_at DESC);
        """
                )
                conn.commit()

    def bulk_insert_orders(self, orders: List[GundamOrder]):
        insert_query = """
        INSERT INTO orders (
            order_id,
            order_detail,
            total_price,
            customer_name,
            customer_email,
            status,
            created_at,
            updated_at
        )
        VALUES %s
    """
        try:
            with psycopg.connect(
                "postgresql://postgres:postgres@localhost:5432/postgres"
            ) as conn:
                with conn.cursor() as cur:
                    order_data = [order.to_tuple() for order in orders]
                    cur.executemany(
                        insert_query.replace("%s", "(%s, %s, %s, %s, %s, %s, %s, %s)"),
                        order_data,
                    )
                    rows_inserted = cur.rowcount
                    return rows_inserted
        except Exception as e:
            logger.error(f"Error inserting orders: {e}")
            raise e


def main():
    num_orders = 50
    random_orders = generate_random_orders(num_orders)
    pg_orders = PGOrders()

    try:
        inserted_count = pg_orders.bulk_insert_orders(random_orders)
        print(f"Inserted {inserted_count} orders into the database")
    except Exception as e:
        logger.error(f"Error inserting orders: {e}")


if __name__ == "__main__":
    main()
