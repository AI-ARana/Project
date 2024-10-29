import pandas as pd

# Creating a dictionary to structure the data for Excel
invoice_data = {
    "Invoice Details": [
        {"Field": "Invoice No", "Value": "24-25/01"},
        {"Field": "Date", "Value": "24-Oct-24"},
        {"Field": "Customer Name", "Value": "Foundation For Life Sciences And Business Management"},
        {"Field": "Address", "Value": "Anand Campus, The Mall, Solan, Himachal Pradesh-173212"}
    ],
    "Training Details": [
        {"Field": "Description", "Value": "Training Charges for Machine Learning and Foundations of Data Science Training"},
        {"Field": "Training Period", "Value": "From 20th August 2024 to 30th September 2024"},
        {"Field": "Duration", "Value": "1 Month & 12 Days"},
        {"Field": "Unit Price Per Month", "Value": "₹1,65,000.00"},
        {"Field": "Total", "Value": "₹2,31,000.00"},
        {"Field": "Amount in Words", "Value": "Rs. Two Lakh Thirty One Thousand Only"}
    ],
    "Bank Details": [
        {"Field": "Account Name", "Value": "Aquib Rashid"},
        {"Field": "Bank Name", "Value": "Bank of India"},
        {"Field": "Account Number", "Value": "459010310000117"},
        {"Field": "IFSC Code", "Value": "BKID0004590"},
        {"Field": "Branch", "Value": "Rafiganj"}
    ]
}

# Converting the dictionary to a DataFrame
invoice_df = pd.concat({key: pd.DataFrame(value) for key, value in invoice_data.items()}, names=['Section', 'Index']).reset_index(level=0)

# Create the Excel file
excel_path = Invoice_Details.xlsx"
invoice_df.to_excel(excel_path, index=False, sheet_name="Invoice Details")

excel_path
