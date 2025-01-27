from flask import Flask, request, jsonify
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import pandas as pd
import pytesseract
import cv2
import pymssql

app = Flask(__name__)

# Initialize model and processor
model = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2-VL-2B-Instruct-AWQ", torch_dtype="auto")
if torch.cuda.is_available():
    model.to("cuda")

processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct-AWQ")
pytesseract.pytesseract_cmd = r'/usr/bin/tesseract'

# Function to identify category based on keywords
def identify_category(text):
    text = text.lower()
    if any(keyword in text for keyword in ["food", "meal", "restaurant", "cafe", "coffee", "drink"]):
        return "Food"
    elif any(keyword in text for keyword in ["travel", "flight", "bus", "car", "taxi", "train", "ticket"]):
        return "Travel"
    elif any(keyword in text for keyword in ["hotel", "stay", "room", "resort", "accommodation"]):
        return "Stay"
    else:
        return "Others"

# Store DataFrame to Azure SQL Database
def store_to_azure_sql(dataframe):
    try:
        conn = pymssql.connect(
            server="piosqlserverbd.database.windows.net",
            user="pio-admin",
            password="Poctest123#",
            database="PIOSqlDB"
        )
        cursor = conn.cursor()

        create_table_query = """
        IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='Invoices' AND xtype='U')
        CREATE TABLE Invoices (
            EmployeeID NVARCHAR(50) NOT NULL PRIMARY KEY,
            InvoiceNumber NVARCHAR(255),
            Date NVARCHAR(255),
            Place NVARCHAR(255),
            Amount NVARCHAR(255),
            Category NVARCHAR(255),
            ApprovalStatus NVARCHAR(50) DEFAULT 'Pending'
        )
        """
        cursor.execute(create_table_query)

        cursor.execute("SELECT TOP 1 EmployeeID FROM Invoices ORDER BY EmployeeID DESC")
        last_id = cursor.fetchone()
        next_id = 0 if last_id is None else int(last_id[0]) + 1

        for _, row in dataframe.iterrows():
            category = identify_category(row["Invoice Details"])
            insert_query = """
            INSERT INTO Invoices (EmployeeID, InvoiceNumber, Date, Place, Amount, Category, ApprovalStatus)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """
            cursor.execute(
                insert_query,
                (
                    f"{next_id:03d}",
                    row.get("Invoice Number", "")[:255],
                    row.get("Date", ""),
                    row.get("Place", ""),
                    row.get("Amount", ""),
                    category,
                    "Pending"
                )
            )
            next_id += 1

        conn.commit()
        conn.close()
        return "Data successfully stored in Azure SQL Database."
    except Exception as e:
        return f"Error storing data to database: {e}"

# Process image and extract details
def process_image(image_path):
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": (
                "Extract the following details from the invoice:\n"
                "- 'invoice_number'\n"
                "- 'date'\n"
                "- 'place'\n"
                "- 'amount' (monetary value in the relevant currency)\n"
                "- 'category' (based on the invoice type)"
            )}
        ]
    }]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    inputs = inputs.to(model.device)
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return parse_details(output_text[0])

def parse_details(details):
    parsed_data = {
        "Invoice Number": None,
        "Date": None,
        "Place": None,
        "Amount": None,
        "Invoice Details": details
    }
    lines = details.split("\n")
    for line in lines:
        lower_line = line.lower()
        if "invoice" in lower_line:
            parsed_data["Invoice Number"] = line.split(":")[-1].strip()
        elif "date" in lower_line:
            parsed_data["Date"] = line.split(":")[-1].strip()
        elif "place" in lower_line:
            parsed_data["Place"] = line.split(":")[-1].strip()
        elif any(keyword in lower_line for keyword in ["total", "amount", "cost"]):
            parsed_data["Amount"] = line.split(":")[-1].strip()
    return parsed_data

@app.route('/extract', methods=['POST'])
def extract_invoice():
    image_path = request.json.get('image_path')
    extracted_data = process_image(image_path)
    df = pd.DataFrame([extracted_data])
    status = store_to_azure_sql(df)
    return jsonify({"data": extracted_data, "status": status})

if __name__ == '__main__':
    app.run(port=22)
