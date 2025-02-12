import pandas as pd
import numpy as np
from faker import Faker



def generate_data():
    """
        generate synthetic data 
    
    """
    
    
    fake = Faker()

    # Define possible banking products
    banking_products = ["Savings Account", "Current Account", "Credit Card", "Personal Loan", 
                        "Mortgage Loan", "Fixed Deposit", "Mutual Fund", "Car Loan"]

    # Generate synthetic customer transactions
    num_customers = 500  # Simulating 500 customers
    num_transactions = 5000  # Simulating 5000 transactions

    data = []
    for _ in range(num_transactions):
        customer_id = np.random.randint(1000, 2000)  # Unique customer IDs
        transaction_date = fake.date_between(start_date="-2y", end_date="today")
        product_used = np.random.choice(banking_products)
        transaction_amount = round(np.random.uniform(500, 100000), 2)  # Random amount
        tenure_years = np.random.randint(1, 15)  # How long customer has been with bank
        transaction_frequency = np.random.randint(1, 20)  # Monthly transaction count

        data.append([customer_id, transaction_date, product_used, transaction_amount, tenure_years, transaction_frequency])

    # Create DataFrame
    df = pd.DataFrame(data, columns=["customer_id", "transaction_date", "product_used", 
                                    "transaction_amount", "customer_tenure", "transaction_frequency"])

    # Save dataset as CSV
    df.to_csv("synthetic_banking_transactions.csv", index=False)

    print("✅ Synthetic banking dataset created!")
    print(df.head())

    #3#############################################################


    # Load transaction data
    df = pd.read_csv("")

    # Convert transaction dates
    df["transaction_date"] = pd.to_datetime(df["transaction_date"])
    df = df.sort_values(by=["customer_id", "transaction_date"])

    # Group transactions into ordered sequences
    sequences = df.groupby("customer_id")["product_used"].apply(list).reset_index()

    print("Data ready for sequential modeling!")
