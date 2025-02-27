from fastapi import FastAPI, Request
from langchain_groq import ChatGroq 
from googletrans import Translator
from pymongo import MongoClient
import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import asyncio
import requests


# Initialize FastAPI
app = FastAPI()

# Initialize Translator
translator = Translator()

# Connect to MongoDB (for request classification & knowledge base)

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017")
db = client["chatbot_db"]
requests_collection = db["requests"]

# List of user requests

user_requests = [
    {"user_id": "1", "request": "Check my account balance"},
    {"user_id": "2", "request": "Transfer $500 to John"},
    {"user_id": "3", "request": "Fetch my transaction history"},
    {"user_id": "4", "request": "Update my email address"},
    {"user_id": "5", "request": "Close my savings account"},
    {"user_id": "6", "request": "Reset my password"},
    {"user_id": "7", "request": "Change my phone number"},
    {"user_id": "8", "request": "Get loan eligibility"},
    {"user_id": "9", "request": "Pay my electricity bill"},
    {"user_id": "10", "request": "Book a flight ticket"},
    {"user_id": "11", "request": "Subscribe to premium services"},
    {"user_id": "12", "request": "Download my tax statement"},
    {"user_id": "13", "request": "Enable two-factor authentication"},
    {"user_id": "14", "request": "Find nearby ATMs"},
    {"user_id": "15", "request": "Generate a new card PIN"},
    {"user_id": "16", "request": "Block my lost credit card"},
    {"user_id": "17", "request": "Apply for a new credit card"},
    {"user_id": "18", "request": "Redeem my reward points"},
    {"user_id": "19", "request": "Get a mortgage loan"},
    {"user_id": "20", "request": "Find my nearest bank branch"},
    {"user_id": "21", "request": "Update my home address"},
    {"user_id": "22", "request": "Check my insurance policy status"},
    {"user_id": "23", "request": "Schedule an appointment with an advisor"},
    {"user_id": "24", "request": "Generate my monthly expense report"},
    {"user_id": "25", "request": "Check stock market trends"},
    {"user_id": "26", "request": "Buy shares of Tesla"},
    {"user_id": "27", "request": "Sell my Apple stocks"},
    {"user_id": "28", "request": "Find real estate investment options"},
    {"user_id": "29", "request": "Pay my credit card bill"},
    {"user_id": "30", "request": "Activate international transactions"},
    {"user_id": "31", "request": "Enable mobile banking"},
    {"user_id": "32", "request": "Get my credit score"},
    {"user_id": "33", "request": "Request for a higher credit limit"},
    {"user_id": "34", "request": "Upgrade my bank account"},
    {"user_id": "35", "request": "Download my investment portfolio report"},
    {"user_id": "36", "request": "Find the best savings account interest rates"},
    {"user_id": "37", "request": "Invest in cryptocurrency"},
    {"user_id": "38", "request": "Retrieve my forgotten security questions"},
    {"user_id": "39", "request": "Report a fraudulent transaction"},
    {"user_id": "40", "request": "Set up automatic bill payments"},
    {"user_id": "41", "request": "Cancel my subscription"},
    {"user_id": "42", "request": "Find the nearest car loan options"},
    {"user_id": "43", "request": "Request a bank statement for last 6 months"},
    {"user_id": "44", "request": "Unlock my locked account"},
    {"user_id": "45", "request": "Apply for student loan"},
    {"user_id": "46", "request": "Check forex exchange rates"},
    {"user_id": "47", "request": "Withdraw funds from my fixed deposit"},
    {"user_id": "48", "request": "View my savings account balance"},
    {"user_id": "49", "request": "Order a new cheque book"},
    {"user_id": "50", "request": "Change my internet banking password"},
    {"user_id": "51", "request": "Track my loan repayment schedule"},
    {"user_id": "52", "request": "Check the status of my tax refund"},
    {"user_id": "53", "request": "Enable auto investment"},
    {"user_id": "54", "request": "Donate to charity"},
    {"user_id": "55", "request": "Apply for a personal loan"},
    {"user_id": "56", "request": "Generate a credit card transaction report"},
    {"user_id": "57", "request": "Set up a recurring deposit"},
    {"user_id": "58", "request": "View my upcoming bill payments"},
    {"user_id": "59", "request": "Get travel insurance"},
    {"user_id": "60", "request": "Retrieve old banking statements"},
    {"user_id": "61", "request": "Request a tax invoice"},
    {"user_id": "62", "request": "Find retirement investment plans"},
    {"user_id": "63", "request": "Order a replacement debit card"},
    {"user_id": "64", "request": "Change my beneficiary details"},
    {"user_id": "65", "request": "Get investment recommendations"},
    {"user_id": "66", "request": "Apply for business financing"},
    {"user_id": "67", "request": "Find tax-saving investment options"},
    {"user_id": "68", "request": "Lock my card for international use"},
    {"user_id": "69", "request": "Set up an emergency savings fund"},
    {"user_id": "70", "request": "Activate a standing instruction"},
    {"user_id": "71", "request": "Retrieve my loan agreement copy"},
    {"user_id": "72", "request": "Check my card cashback rewards"},
    {"user_id": "73", "request": "Set a savings goal"},
    {"user_id": "74", "request": "Convert my salary account to premium"},
    {"user_id": "75", "request": "Apply for home insurance"},
    {"user_id": "76", "request": "Renew my car insurance policy"},
    {"user_id": "77", "request": "Get a detailed transaction summary"},
    {"user_id": "78", "request": "Register a complaint about my bank service"},
    {"user_id": "79", "request": "Request a reversal for a failed transaction"},
    {"user_id": "80", "request": "Find the best credit card offers"},
    {"user_id": "81", "request": "View my mutual fund investments"},
    {"user_id": "82", "request": "Modify my standing instruction"},
    {"user_id": "83", "request": "Check if my cheque has cleared"},
    {"user_id": "84", "request": "Download my e-passbook"},
    {"user_id": "85", "request": "Enable instant fund transfer"},
    {"user_id": "86", "request": "Report an unauthorized transaction"},
    {"user_id": "87", "request": "View my banking service charges"},
    {"user_id": "88", "request": "Check my EMI due date"},
    {"user_id": "89", "request": "Find offers on personal loans"},
    {"user_id": "90", "request": "Upgrade my credit card plan"},
    {"user_id": "91", "request": "Modify my account nominee"},
    {"user_id": "92", "request": "View my bill payment history"},
    {"user_id": "93", "request": "Find loan prepayment charges"},
    {"user_id": "94", "request": "Get notified about new investment opportunities"},
    {"user_id": "95", "request": "Change my debit card spending limit"},
    {"user_id": "96", "request": "Update my KYC details"},
    {"user_id": "97", "request": "Enable biometric authentication"},
    {"user_id": "98", "request": "Apply for overdraft protection"},
    {"user_id": "99", "request": "Download my home loan statement"},
    {"user_id": "100", "request": "Check my card reward redemption status"}
]

#Insert into MongoDB
requests_collection.insert_many(user_requests)
knowledge_collection = db["knowledge"]

# OpenAI/Groq API Key
GROQ_API_KEY = "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"  # Replace with your actual API key

llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama-3.3-70b-specdec",  # Choose a supported model
    temperature=0.7
)

# Load NLP Model for Classification
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize FAISS index for request similarity search
request_embeddings = []
request_texts = []
for request in requests_collection.find():
    request_texts.append(request["request"])
    request_embeddings.append(embedding_model.encode(request["request"]))

if request_embeddings:
    index = faiss.IndexFlatL2(len(request_embeddings[0]))
    index.add(np.array(request_embeddings))
else:
    index = None

# Function to classify user requests
def classify_request(user_input):
    if not index:
        return "Unknown service"
    user_embedding = embedding_model.encode(user_input)
    D, I = index.search(np.array([user_embedding]), k=1)
    return request_texts[I[0][0]] if D[0][0] < 0.7 else "Unknown service"

# Function to get AI-generated responses with knowledge retrieval
def get_ai_response(user_input):
    response = llm.invoke([{"role": "user", "content": user_input}])
    return response.content


# Function to handle multilingual support
async def translate_text(text, target_lang):
    translation = await translator.translate(text, dest=target_lang)
    return translation.text

@app.post("/chat")
async def chat_endpoint(request: Request):
    data = await request.json()
    user_input = data.get("message", "")
    translated_input = await translate_text(user_input, "en")  # Await the function

    # Check if request needs classification
    service = classify_request(translated_input)
    if service != "Unknown service":
        response = f"Your request has been routed to {service}."
    else:
        response = get_ai_response(translated_input)
    
    translated_response = await translate_text(response, "pt")  # Await the function
    return {"response": translated_response}


# Streamlit UI
st.title("Oráculo Bot - Intelligent Resident Chatbot")
user_input = st.text_input("Digite sua pergunta (em português)")


if st.button("Enviar"):
    response = requests.post("http://127.0.0.1:8000/chat", json={"message": user_input})
    if response.status_code == 200:
        st.write(response.json()["response"])
    else:
        st.write("Erro ao obter resposta do chatbot.")
