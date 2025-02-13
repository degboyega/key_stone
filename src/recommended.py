import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from openai import OpenAI

# Define banking products with descriptions
BANKING_PRODUCTS = [
    {
        "name": "basic_checking_account",
        "description": "A no-frills checking account with low fees.",
        "category": "account",
    },
    {
        "name": "premium_checking_account",
        "description": "A high-tier checking account with benefits like cashback and no ATM fees.",
        "category": "account",
    },
    {
        "name": "savings_account",
        "description": "A standard savings account with competitive interest rates.",
        "category": "account",
    },
    {
        "name": "high_yield_savings_account",
        "description": "A savings account with higher interest rates for larger balances.",
        "category": "account",
    },
    {
        "name": "credit_card",
        "description": "A standard credit card with rewards on everyday purchases.",
        "category": "credit",
    },
    {
        "name": "platinum_credit_card",
        "description": "A premium credit card with travel rewards and concierge services.",
        "category": "credit",
    },
    {
        "name": "personal_loan",
        "description": "A loan for personal expenses with flexible repayment terms.",
        "category": "loan",
    },
    {
        "name": "low_interest_loan",
        "description": "A loan with lower interest rates for qualified customers.",
        "category": "loan",
    },
    {
        "name": "investment_account",
        "description": "An account for investing in stocks, bonds, and mutual funds.",
        "category": "investment",
    },
    {
        "name": "retirement_account",
        "description": "A tax-advantaged account for retirement savings.",
        "category": "investment",
    },
]


# Generate synthetic transaction data
def generate_synthetic_transaction_data(
    num_customers=100, transactions_per_customer=50
):
    np.random.seed(42)

    # Generate customer IDs
    customer_ids = np.arange(1, num_customers + 1)

    # Generate synthetic data
    data = []
    for customer_id in customer_ids:
        # Customer-specific attributes
        tenure = np.random.randint(1, 10)  # Customer tenure in years
        transaction_freq = np.random.poisson(20)  # Average transactions per month

        # Generate transactions
        for _ in range(transactions_per_customer):
            timestamp = datetime.now() - timedelta(
                days=np.random.randint(0, 365 * tenure)
            )
            transaction_type = np.random.choice(
                ["deposit", "withdrawal", "transfer", "payment"]
            )
            transaction_amount = np.abs(np.random.normal(100, 50))  # Random amount
            product_used = np.random.choice(
                [product["name"] for product in BANKING_PRODUCTS]
            )

            data.append(
                [
                    customer_id,
                    timestamp,
                    transaction_type,
                    round(transaction_amount, 2),
                    product_used,
                    tenure,
                    transaction_freq,
                ]
            )

    # Create DataFrame
    columns = [
        "customer_ID",
        "timestamp",
        "transaction_type",
        "transaction_amount",
        "product_used",
        "customer_tenure",
        "transaction_frequency",
    ]
    df = pd.DataFrame(data, columns=columns)
    return df


# Preprocess data for collaborative filtering
def preprocess_collaborative_filtering(transaction_data):
    # Create a customer-product matrix
    customer_product_matrix = transaction_data.pivot_table(
        index="customer_ID",
        columns="product_used",
        values="transaction_amount",
        aggfunc="count",
        fill_value=0,
    )
    return customer_product_matrix


# Collaborative filtering recommendations
def collaborative_filtering_recommendations(
    customer_id, customer_product_matrix, top_n=3
):
    # Compute similarity between customers
    similarity_matrix = cosine_similarity(customer_product_matrix)
    similarity_df = pd.DataFrame(
        similarity_matrix,
        index=customer_product_matrix.index,
        columns=customer_product_matrix.index,
    )

    # Find similar customers
    similar_customers = (
        similarity_df[customer_id].sort_values(ascending=False).index[1:]
    )

    # Get products used by similar customers
    recommended_products = (
        customer_product_matrix.loc[similar_customers]
        .sum()
        .sort_values(ascending=False)
        .index
    )

    # Filter out products already used by the customer
    used_products = customer_product_matrix.loc[customer_id][
        customer_product_matrix.loc[customer_id] > 0
    ].index
    recommended_products = [
        product for product in recommended_products if product not in used_products
    ]

    return recommended_products[:top_n]


# Handle cold start (new customers)
def cold_start_recommendations(customer_data, top_n=3):
    # Recommend popular products
    popular_products = customer_data["product_used"].value_counts().index
    return popular_products[:top_n]


# Generate personalized message using LLM
def generate_personalized_message(
    customer_data, recommended_products, openai_api_key=None
):
    if not openai_api_key:
        return f"Sample personalized message: We recommend {', '.join(recommended_products)} based on your transaction history."

    client = OpenAI(api_key=openai_api_key)

    # Prepare prompt
    prompt = f"""
    Generate a personalized banking recommendation message for a customer with these characteristics:
    - Most Used Product: {customer_data['product_used'].mode()[0]}
    - Transaction Frequency: {customer_data['transaction_frequency'].mean():.1f} transactions/month
    - Customer Tenure: {customer_data['customer_tenure'].mean():.1f} years
    
    Recommended Products: {', '.join(recommended_products)}
    
    Create a friendly, professional message that explains why these products are recommended.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful banking assistant."},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Could not generate message: {str(e)}"


# Generate recommendations for a subset of customers
def generate_recommendations_for_subset(
    transaction_data, customer_product_matrix, num_customers, openai_api_key=None
):
    results = []

    # Get the first `num_customers` customers
    customer_ids = transaction_data["customer_ID"].unique()[:num_customers]

    for customer_id in customer_ids:
        selected_customer = transaction_data[
            transaction_data["customer_ID"] == customer_id
        ]

        # Generate recommendations
        if customer_id in customer_product_matrix.index:
            recommended_products = collaborative_filtering_recommendations(
                customer_id, customer_product_matrix
            )
        else:
            recommended_products = cold_start_recommendations(transaction_data)

        # Generate personalized message
        message = generate_personalized_message(
            selected_customer, recommended_products, openai_api_key
        )

        # Create customer summary profile
        summary_profile = f"""
        Tenure: {selected_customer['customer_tenure'].mean():.1f} years
        Transaction Frequency: {selected_customer['transaction_frequency'].mean():.1f}/month
        Most Used Product: {selected_customer['product_used'].mode()[0]}
        """

        # Append results
        results.append(
            {
                "Customer ID": customer_id,
                "Customer Summary Profile": summary_profile,
                "Recommended Products": ", ".join(recommended_products),
                "Personalized Message": message,
            }
        )

    return pd.DataFrame(results)


# Streamlit UI
def main():
    st.title("Personalized Banking Recommendations")

    # Generate synthetic transaction data
    transaction_data = generate_synthetic_transaction_data()

    # Preprocess data for collaborative filtering
    customer_product_matrix = preprocess_collaborative_filtering(transaction_data)

    # Sidebar for customer selection and number of customers
    st.sidebar.header("Settings")
    customer_id = st.sidebar.selectbox(
        "Select Customer ID", transaction_data["customer_ID"].unique()
    )

    num_customers = st.sidebar.slider(
        "Number of Customers to Generate Recommendations For",
        min_value=1,
        max_value=len(transaction_data["customer_ID"].unique()),
        value=10,  # Default value
    )

    selected_customer = transaction_data[transaction_data["customer_ID"] == customer_id]

    # Display customer profile
    st.header("Customer Profile")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Customer Tenure",
            f"{selected_customer['customer_tenure'].mean():.1f} years",
        )
    with col2:
        st.metric(
            "Transaction Frequency",
            f"{selected_customer['transaction_frequency'].mean():.1f}/month",
        )
    with col3:
        st.metric("Most Used Product", selected_customer["product_used"].mode()[0])

    st.subheader("Transaction History (Last 10 Transactions)")
    st.write(
        selected_customer[
            ["timestamp", "transaction_type", "transaction_amount", "product_used"]
        ].head(10)
    )

    # Generate recommendations for the selected customer
    if customer_id in customer_product_matrix.index:
        recommended_products = collaborative_filtering_recommendations(
            customer_id, customer_product_matrix
        )
    else:
        recommended_products = cold_start_recommendations(transaction_data)

    st.header("Recommended Products")
    for product in recommended_products:
        st.success(f"✓ {product}")

    # Generate personalized message
    st.header("Personalized Message")
    # openai_api_key = st.text_input("Enter OpenAI API Key (optional)", type="password")
    openai_api_key = None

    message = generate_personalized_message(
        selected_customer, recommended_products, openai_api_key
    )
    st.write(message)

    # Generate recommendations for a subset of customers
    if st.button("Generate Recommendations for Selected Number of Customers"):
        st.header(f"Recommendations for {num_customers} Customers")
        subset_recommendations_df = generate_recommendations_for_subset(
            transaction_data, customer_product_matrix, num_customers, openai_api_key
        )
        st.dataframe(subset_recommendations_df)

    # Show raw data
    if st.checkbox("Show raw transaction data"):
        st.write(selected_customer)


if __name__ == "__main__":
    main()
