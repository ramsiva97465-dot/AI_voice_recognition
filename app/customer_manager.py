import json
import re
from pathlib import Path
from datetime import datetime

CUSTOMERS_FILE = Path("database") / "customers.json"

def _ensure_file_exists():
    """Ensure the customers.json file and its parent directories exist."""
    CUSTOMERS_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not CUSTOMERS_FILE.exists():
        with open(CUSTOMERS_FILE, "w") as f:
            json.dump([], f)

def load_customers() -> list:
    """Load the list of customers from the JSON file."""
    _ensure_file_exists()
    try:
        with open(CUSTOMERS_FILE, "r") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return []

def save_customers(customers: list):
    """Save the list of customers to the JSON file."""
    _ensure_file_exists()
    with open(CUSTOMERS_FILE, "w") as f:
        json.dump(customers, f, indent=4)

def generate_customer_id() -> str:
    """Generate the next sequential customer ID (e.g., CUST000001)."""
    customers = load_customers()
    if not customers:
        return "CUST000001"
    
    max_id = 0
    pattern = re.compile(r"^CUST(\d{6})$")
    
    for customer in customers:
        match = pattern.match(customer.get("customer_id", ""))
        if match:
            num = int(match.group(1))
            if num > max_id:
                max_id = num
                
    next_id = max_id + 1
    return f"CUST{next_id:06d}"

def create_customer(customer_id: str) -> dict:
    """Create a new customer record."""
    customers = load_customers()
    
    now_iso = datetime.now().replace(microsecond=0).isoformat()
    new_customer = {
        "customer_id": customer_id,
        "created_at": now_iso,
        "last_call": now_iso,
        "call_count": 1
    }
    
    customers.append(new_customer)
    save_customers(customers)
    return new_customer

def update_customer(customer_id: str) -> dict:
    """Update an existing customer's last_call and increment call_count."""
    customers = load_customers()
    now_iso = datetime.now().replace(microsecond=0).isoformat()
    
    for customer in customers:
        if customer.get("customer_id") == customer_id:
            customer["last_call"] = now_iso
            customer["call_count"] = customer.get("call_count", 0) + 1
            save_customers(customers)
            return customer
            
    # If not found for some reason, create it to self-heal
    return create_customer(customer_id)

def get_customer(customer_id: str) -> dict:
    """Retrieve customer data by ID."""
    customers = load_customers()
    for customer in customers:
        if customer.get("customer_id") == customer_id:
            return customer
    return {}
