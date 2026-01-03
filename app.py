import pandas as pd
import matplotlib.pyplot as plt
from openai import OpenAI
import streamlit as st
import os

# Load data
df = pd.read_csv("archive/expense_data_1.csv")
print(df.head())

# clean data- keep only relevant columns
data = df[["Date", "Category", "Note", "Amount", "Income/Expense"]]
print(data.head())

def add_expense(date, category, note, amount, exp_type="Expense"):
    global data
    new_entry = {
        "Date": date,
        "Category": category,
        "Note": note,
        "Amount": amount,
        "Income/Expense": exp_type
    }
    data = pd.concat([data, pd.DataFrame([new_entry])], ignore_index=True)
    print(f" Added: {note} - {amount} ({category})")

add_expense("2025-12-22 19:30", "Food", "Shawarma", 2500, "Expense")
add_expense("2025-12-23 08:00", "Subscriptions", "Netflix Monthly Plan", 4500, "Expense")
add_expense("2025-12-24 14:00", "Entertainment", "Outdoor Games with friends", 7000, "Expense")

def view_expenses(n=5):
    return data.tail(n)
print(view_expenses(5))

def summarize_expenses(by="Category"):
    summary = data[data["Income/Expense"]=="Expense"].groupby(by)["Amount"].sum()
    return summary.sort_values(ascending=False)
print(summarize_expenses())


MODEL = "llama3.2"
client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

# We use an LLM to read the Note column 
# and then automatically assign the most relevant Category.

def auto_categorize(note):
    prompt = f"""
    Categorize this expense note into one of these categories: 
    Food, Transportation, Entertainment, Other.
    Note: {note}
    """
    try:
        response = client.chat.completions.create(
            model= MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return "Other"

data['Category'] = data.apply(
    lambda row: auto_categorize(row['Note']) if pd.isna(row['Category']) else row['Category'],
    axis=1
)
print(data[['Note', 'Category']].head(10))

# Visualize expenses
expense_summary = data[data['Category'] != 'Income'].groupby("Category")["Amount"].sum()

# Pie Chart
plt.figure(figsize=(6,6))
expense_summary.plot.pie(autopct='%1.1f%%', startangle=90, shadow=True)
plt.title("Expenses Breakdown by Category")
plt.ylabel("")
plt.show()

# Bar Chart
plt.figure(figsize=(8,5))
expense_summary.plot(kind="bar", color="skyblue", edgecolor="black")
plt.title("Expenses by Category")
plt.xlabel("Category")
plt.ylabel("Amount Spent")
plt.show()

def predict_category(description):
    prompt = f"""
    You are a financial assistant. Categorize this expense into one of:
    ['Food', 'Transportation', 'Entertainment', 'Utilities', 'Shopping', 'Others'].

    Expense: "{description}"
    Just return the category name.
    """
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content.strip()

csv_file = "archive/expense_data_1.csv"
if os.path.exists(csv_file):
    data = pd.read_csv(csv_file)
else:
    data = pd.DataFrame(columns=["Date", "Description", "Amount", "Category"])

st.title("Smart Expense Tracker")

with st.form("expense_form"):
    date = st.date_input("Date")
    description = st.text_input("Description")
    amount = st.number_input("Amount", min_value=0.0, format="%.2f")

    predicted_category = ""
    if description:
        predicted_category = predict_category(description)

    category = st.text_input(
        "Category (auto-predicted, but you can edit)", 
        value=predicted_category
    )

    submitted = st.form_submit_button("Add Expense")

    if submitted:
        new_expense = {"Date": date, "Description": description, "Amount": amount, "Category": category}
        data = pd.concat([data, pd.DataFrame([new_expense])], ignore_index=True)
        data.to_csv(csv_file, index=False)
        st.success(f"Added: {description} - {amount} ({category})")

st.subheader("All Expenses")
st.dataframe(data)

if not data.empty:
    st.subheader("Expense Breakdown by Category")

    category_totals = data.groupby("Category")["Amount"].sum()

    # Bar Chart
    fig, ax = plt.subplots()
    category_totals.plot(kind="bar", ax=ax)
    ax.set_ylabel("Amount")
    st.pyplot(fig)

    # Pie Chart
    st.subheader("Category Distribution")
    fig2, ax2 = plt.subplots()
    category_totals.plot(kind="pie", autopct="%1.1f%%", ax=ax2)
    st.pyplot(fig2)


