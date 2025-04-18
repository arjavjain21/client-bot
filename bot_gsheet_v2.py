import os
import json
import re
import csv
import logging
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# -------------------------------
# Logging: Console & File Handler
# -------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# Create handlers
console_handler = logging.StreamHandler()
file_handler = logging.FileHandler("bot.log")
# Create formatter and add it to handlers
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)
# Add handlers to logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Environment variables for Slack, OpenAI, CSV, and Google Sheets
SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN")
SLACK_APP_TOKEN = os.environ.get("SLACK_APP_TOKEN")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
CSV_FILE_PATH = os.environ.get("CSV_FILE_PATH", "clients.csv")
GOOGLE_SHEET_ID = os.environ.get("GOOGLE_SHEET_ID")
SHEET_NAME = os.environ.get("SHEET_NAME")
GOOGLE_CREDENTIALS_JSON = os.environ.get("GOOGLE_CREDENTIALS_JSON")

# Set OpenAI API key
import openai
openai.api_key = OPENAI_API_KEY

# Global conversation memory and query cache
conversation_memory = {}  # user_id -> list of {"role": "user"/"assistant"/"system", "content": "…"}
query_cache = {}

# Global document store and FAISS index
documents = []         # List of client document strings
doc_index = None       # FAISS index
doc_embeddings = None  # Numpy array of embeddings

# -------------------------------
# Extended Header Mapping Dictionary
# -------------------------------
HEADER_MAP = {
    "Client Email": "client_email",
    "Account Code": "account_code",
    "Your Company's Name": "company_name",
    "What countries/geographies are your prospects located in?": "prospect_geography",
    "What is your ideal prospect's title (prospect = the person we will email)?": "prospect_title",
    "What industries are the prospects' companies in?": "prospect_industries",
    "Do you qualify prospects by employee count or revenue? If so, what are the ideal employee/revenue ranges for the companies you wish to target?": "qualification_ranges",
    "Any other qualifiers that separate qualified prospects from unqualified ones?": "qualifiers",
    "What is the service (solution mechanism) that you offer? In case of multiple services, please list all of them.": "service_offered",
    "Please state your offer - the result you produce or the problem you solve - in quantifiable terms with timeframes. This is the result you are confident in delivering for the PERFECT customer.\n\n(ex: We help e-commerce brands add $150k-$1m in revenue, without extra ad-spend within 6 months using email and SMS marketing)": "offer_description",
    "What is the USP of your offer - what makes prospects choose you instead of going to competitors, in-house teams, or simply doing nothing?": "offer_usp",
    "Please share past results and case studies. Please do not share PDFs or website links. Simple sentences in the following format will work:\n\nWe helped <client X> do <Y positive outcome> within <Z timeframe> using/without A, B, C.\n\nWrite as many examples as you can (at least 3).": "case_studies",
    "Which of the above clients' names would you NOT want to reveal when doing outreach?": "do_not_reveal",
    "Did you win any awards/certifications/other recognitions doing what you do?": "awards",
    "Please link below any VSLs, training videos, case study videos or other sales assets that you've created in the past and that you are willing to offer to prospects as a hook to get them to book a call\n\nCase Study 1 XYZ result: <link>\nVideo training 1: <link>": "sales_assets",
    "In your experience from past sales calls, what else makes prospects want to work with you?": "sales_hooks",
    "Please add a Google Sheet link to your DO NOT CONTACT list containing the emails and websites of existing clients or prospects that you don't want us to contact.\n(In case of no input, we assume you're okay with us reaching out to all businesses)": "do_not_contact_list",
    "placeholder_for_pabbly": "pabbly_placeholder",
    "Timestamp": "timestamp",
    "What are the top 3 challenges that they are trying to overcome right now? (answer to the best of your knowledge)": "top_challenges",
    "What tools/vendors/other solutions have they already tried?": "tools_tried",
    "How many Leads/Appointments would you Reasonably want to see every month for this effort to be a success in your eyes?": "leads_per_month",
    "Please list your top 10 direct competitors - 5 that are industry leaders, 3 that are your size, 2 that": "top_competitors",
    "How is your approach better than each of the solution mechanisms in your response above?": "approach_better_than",
    "Please list at least 6 direct competitors - min. 2 industry leaders, min. 2 companies comparable to you in size, and, min. 2 that you your customers might be using instead of using you": "direct_competitors",
    "Please list at least 6 companies that you would LOVE to work with and can realistically service. Feel free to explain why they are a good fit.": "ideal_clients",
    "What do you charge (a price range works)? We use this to answer questions when prospects insist on seeing a price before they agree to a call. We usually give them a range: between $X and $Y.": "price_range",
    "STATUS": "status",
    "Offer for Outreach": "offer_for_outreach",
    "Lead Sources": "lead_sources",
    "Your business email address": "client_email",
    "Your company website": "company_website",
    "Your First Name": "first_name",
    "Your Last Name": "last_name",
    "Your Phone Number": "phone_number"
}

# -------------------------------
# Data Ingestion Functions (with Header Mapping)
# -------------------------------
def load_documents_from_csv(csv_file_path):
    docs = []
    try:
        with open(csv_file_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                doc_lines = []
                for header, value in row.items():
                    simplified_key = HEADER_MAP.get(header, header)
                    if value:
                        doc_lines.append(f"{simplified_key}: {value}")
                document = "\n".join(doc_lines)
                docs.append(document)
        logger.info("Loaded %d documents from CSV.", len(docs))
    except Exception as e:
        logger.error("Error loading CSV: %s", e)
    return docs

def load_documents_from_gsheet():
    try:
        import gspread
        from google.oauth2.service_account import Credentials
        credentials_info = json.loads(GOOGLE_CREDENTIALS_JSON) if GOOGLE_CREDENTIALS_JSON else {}
        if not credentials_info:
            logger.error("Google credentials not provided.")
            return []
        scopes = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
        creds = Credentials.from_service_account_info(credentials_info, scopes=scopes)
        client = gspread.authorize(creds)
        sheet = client.open_by_key(GOOGLE_SHEET_ID).worksheet(SHEET_NAME)
        rows = sheet.get_all_records()
        docs = []
        for row in rows:
            doc_lines = []
            for header, value in row.items():
                simplified_key = HEADER_MAP.get(header, header)
                if value:
                    doc_lines.append(f"{simplified_key}: {value}")
            document = "\n".join(doc_lines)
            docs.append(document)
        logger.info("Loaded %d documents from Google Sheets.", len(docs))
        return docs
    except Exception as e:
        error_msg = f"Error loading documents from Google Sheets: {e}"
        logger.error(error_msg)
        return [f"API Error: {error_msg}"]

def build_index(docs, embedder):
    import numpy as np
    import faiss
    embeddings = embedder.encode(docs, convert_to_tensor=False)
    embeddings = np.array(embeddings).astype('float32')
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    logger.info("Built FAISS index with %d documents.", len(docs))
    return index, embeddings

# -------------------------------
# Incremental Client Data Initialization
# -------------------------------
def initialize_client_data():
    """
    Initialize or incrementally update client data from Google Sheets.
    New documents are added to the existing FAISS index rather than rebuilding it.
    """
    global documents, doc_index, doc_embeddings
    logger.info("Initializing client data from Google Sheets...")

    if GOOGLE_SHEET_ID and SHEET_NAME and GOOGLE_CREDENTIALS_JSON:
        new_docs = load_documents_from_gsheet()
        if not new_docs:
            logger.error("Failed to load documents from Google Sheets.")
            return
    else:
        logger.error("Google Sheets configuration not provided.")
        return

    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    # First-time load: build full index.
    if not documents:
        documents = new_docs
        doc_index, doc_embeddings = build_index(documents, embedder)
        logger.info("Client data initialization complete. Total documents: %d", len(documents))
    else:
        # Check for new documents (assuming new_docs are appended)
        if len(new_docs) > len(documents):
            new_count = len(new_docs) - len(documents)
            logger.info("Found %d new documents. Incrementally updating index.", new_count)
            additional_docs = new_docs[len(documents):]
            import numpy as np
            additional_embeddings = embedder.encode(additional_docs, convert_to_tensor=False)
            additional_embeddings = np.array(additional_embeddings).astype('float32')
            doc_index.add(additional_embeddings)
            documents.extend(additional_docs)
            doc_embeddings = np.concatenate((doc_embeddings, additional_embeddings), axis=0)
        else:
            logger.info("No new documents found during sync.")
        logger.info("Client data update complete. Total documents: %d", len(documents))

def sync_client_data():
    try:
        initialize_client_data()
        return "✅ Data synced successfully from Google Sheets."
    except Exception as e:
        error_msg = f"Error syncing data: {e}"
        logger.error(error_msg)
        return f"Error syncing data: {error_msg}"

# -------------------------------
# Google Sheet Update Functionality
# -------------------------------
def update_gsheet_row(client_email, account_code, updates):
    try:
        import gspread
        from google.oauth2.service_account import Credentials

        credentials_info = json.loads(GOOGLE_CREDENTIALS_JSON) if GOOGLE_CREDENTIALS_JSON else {}
        if not credentials_info:
            logger.error("Google credentials not provided.")
            return "Google credentials not provided."
        scopes = ["https://www.googleapis.com/auth/spreadsheets"]
        creds = Credentials.from_service_account_info(credentials_info, scopes=scopes)
        client = gspread.authorize(creds)
        sheet = client.open_by_key(GOOGLE_SHEET_ID).worksheet(SHEET_NAME)
        headers = sheet.row_values(1)

        # Find the email column index
        email_col = None
        for header_name in ["Client Email", "Your business email address"]:
            if header_name in headers:
                email_col = headers.index(header_name) + 1
                break

        # Find the account code column index
        account_code_col = None
        if "Account Code" in headers:
            account_code_col = headers.index("Account Code") + 1

        row_found = None
        if client_email and email_col:
            cell = sheet.find(client_email, in_column=email_col)
            if cell:
                row_found = cell.row

        if not row_found and account_code and account_code_col:
            cell = sheet.find(account_code, in_column=account_code_col)
            if cell:
                row_found = cell.row

        REVERSE_HEADER_MAP = {v: k for k, v in HEADER_MAP.items()}

        if row_found:
            for key, value in updates.items():
                original_header = REVERSE_HEADER_MAP.get(key)
                if not original_header:
                    logger.warning("No original header for update key '%s'.", key)
                    continue
                if original_header in headers:
                    col_index = headers.index(original_header) + 1
                    sheet.update_cell(row_found, col_index, value)
                    logger.info("Updated row %d, column %d (%s) with value: %s", row_found, col_index, original_header, value)
                else:
                    logger.warning("Header '%s' not found; skipping update for key '%s'.", original_header, key)
            return "✅ Google Sheet updated successfully."
        else:
            new_row = []
            for header in headers:
                if header in HEADER_MAP:
                    simplified = HEADER_MAP[header]
                    if simplified in updates:
                        new_row.append(updates[simplified])
                    elif header in ["Client Email", "Your business email address"] and client_email:
                        new_row.append(client_email)
                    elif header == "Account Code" and account_code:
                        new_row.append(account_code)
                    else:
                        new_row.append("")
                else:
                    new_row.append("")
            sheet.append_row(new_row)
            logger.info("No matching client found; appended a new row.")
            return "✅ Google Sheet did not have a matching client; new row added successfully."
    except Exception as e:
        error_msg = f"Error updating Google Sheet: {e}"
        logger.error(error_msg)
        return f"Error updating Google Sheet: {error_msg}"

def process_update_gsheet_natural_language(text):
    try:
        nl_text = text[len("update gsheet:"):].strip()
        prompt = (
            "You are an assistant that extracts update instructions from natural language text. "
            "Given the following text, output a valid JSON object containing update instructions with the following keys if mentioned: "
            "client_email, account_code, company_name, prospect_geography, prospect_title, prospect_industries, "
            "qualification_ranges, qualifiers, service_offered, offer_description, offer_usp, case_studies, "
            "do_not_reveal, awards, sales_assets, sales_hooks, do_not_contact_list, pabbly_placeholder, timestamp, "
            "top_challenges, tools_tried, leads_per_month, top_competitors, approach_better_than, direct_competitors, "
            "ideal_clients, price_range, status, offer_for_outreach, lead_sources, company_website, first_name, last_name, phone_number. "
            "The JSON must include either 'client_email' or 'account_code'. "
            "Here is the text:\n\n" + nl_text + "\n\nOutput only the JSON object."
        )
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=150,
        )
        json_output = response.choices[0].message.content.strip()
        data = json.loads(json_output)
        return data
    except Exception as e:
        logger.error("Error processing natural language update: %s", e)
        return None

def process_update_gsheet_command(text):
    if not text.lower().startswith("update gsheet:"):
        if text.lower().startswith("update ") and not text.lower().startswith("update memory:"):
            text = "update gsheet:" + text[len("update "):]
    update_content = text[len("update gsheet:"):].strip()
    try:
        data = json.loads(update_content)
    except Exception:
        data = process_update_gsheet_natural_language(text)
    if not data:
        return "Failed to parse update instructions."
    client_email = data.get("client_email")
    account_code = data.get("account_code")
    if not client_email and not account_code:
        return "Either client email or account code is required for updating the Google Sheet."
    updates = {k: v for k, v in data.items() if k not in ["client_email", "account_code"]}
    result = update_gsheet_row(client_email, account_code, updates)
    sync_client_data()
    return result

# -------------------------------
# Internet Search Functionality using DuckDuckGo
# -------------------------------
from duckduckgo_search import DDGS

def lookup_internet(query, max_results=3):
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, safesearch='Moderate', max_results=max_results))
        if results:
            snippets = []
            for result in results:
                title = result.get("title", "No Title")
                href = result.get("href", "No Link")
                snippet = result.get("body", "")
                snippets.append(f"{title} - {href}\n{snippet}")
            return "\n\n".join(snippets)
        else:
            return "No search results found."
    except Exception as e:
        error_msg = f"Error during internet lookup: {e}"
        logger.error(error_msg)
        return f"API Error: {error_msg} - Please check your internet connection and try again."

# -------------------------------
# Query Correction Function
# -------------------------------
def correct_query(query):
    prompt = f"Correct any typos in the following query. Do not change proper names or client-specific terms. Return only the corrected query:\n\n{query}"
    messages = [
        {"role": "system", "content": "You are an expert at correcting typos while preserving proper nouns and client-specific terms."},
        {"role": "user", "content": prompt}
    ]
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages,
            temperature=0,
            max_tokens=50,
        )
        corrected = response.choices[0].message.content.strip()
        logger.info("Original query: '%s' -> Corrected query: '%s'", query, corrected)
        return corrected
    except Exception as e:
        logger.error("Error correcting query: %s", e)
        return query

# -------------------------------
# Client Name Extraction
# -------------------------------
def extract_client_name(query):
    match = re.search(r"tell me about\s+([\w']+)", query, re.IGNORECASE)
    if match:
        candidate = match.group(1).lower().strip()
        if candidate.endswith("'s"):
            candidate = candidate[:-2]
        common_words = {"offer", "icp", "client", "information", "details"}
        if candidate in common_words:
            return None
        return candidate
    return None

# -------------------------------
# Helper: Build Context from Documents
# -------------------------------
def build_context_from_docs(docs, max_tokens):
    context = ""
    for doc in docs:
        doc_tokens = len(gpt2_tokenizer.encode(doc, add_special_tokens=False))
        current_tokens = len(gpt2_tokenizer.encode(context, add_special_tokens=False))
        if current_tokens + doc_tokens <= max_tokens:
            context += doc + "\n"
        else:
            remaining = max_tokens - current_tokens
            if remaining > 50:
                doc_truncated_tokens = gpt2_tokenizer.encode(doc, add_special_tokens=False)[:remaining]
                doc_truncated = gpt2_tokenizer.decode(doc_truncated_tokens, skip_special_tokens=True)
                context += doc_truncated + "\n"
            break
    return context

# -------------------------------
# Offline Retrieval & OpenAI Integration
# -------------------------------
from sentence_transformers import SentenceTransformer as STTransformer
query_embedder = STTransformer('all-MiniLM-L6-v2')

from transformers import AutoTokenizer
gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")
gpt2_tokenizer.model_max_length = 8192  # Allow longer sequences

def retrieve_documents(query, top_k=3):
    global doc_index, documents
    if doc_index is None or not documents:
        return []
    query_embedding = query_embedder.encode([query], convert_to_tensor=False)
    import numpy as np
    query_embedding = np.array(query_embedding).astype("float32")
    distances, indices = doc_index.search(query_embedding, top_k)
    retrieved = []
    for idx in indices[0]:
        if idx < len(documents):
            retrieved.append(documents[idx])
    logger.info("Retrieved %d documents for the query.", len(retrieved))
    return retrieved

def openai_inference(prompt, max_tokens=500):
    messages = [
        {"role": "system", "content": (
            "You are a knowledgeable assistant. Using the provided client data, "
            "answer the query clearly and concisely. If the requested information is missing, state so."
        )},
        {"role": "user", "content": prompt}
    ]
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages,
            temperature=0.7,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error("Error from OpenAI API: %s", e)
        return "Error processing your query."

def process_query_offline(query, user_id):
    if query.lower().startswith("search:"):
        search_term = query[len("search:"):].strip()
        internet_results = lookup_internet(search_term)
        return f"Internet Search Results:\n{internet_results}"
    
    corrected_query = correct_query(query)
    if corrected_query in query_cache:
        logger.info("Returning cached result for query: %s", corrected_query)
        return query_cache[corrected_query]
    
    retrieved_docs = retrieve_documents(corrected_query, top_k=5)
    client_name = extract_client_name(corrected_query)
    if client_name:
        filtered_docs = [doc for doc in retrieved_docs if client_name in doc.lower()]
        additional_docs = [doc for doc in documents if client_name in doc.lower()]
        retrieved_docs = list(set(filtered_docs + additional_docs))
        logger.info("Combined and filtered documents for client name '%s'. Total docs: %d", client_name, len(retrieved_docs))
    
    if not retrieved_docs:
        internet_context = lookup_internet(corrected_query)
        context = f"No local data found. Internet Search Results:\n{internet_context}"
    else:
        conv_memory = conversation_memory.get(user_id, [])
        conversation_context = "".join(f"{turn['role']}: {turn['content']}\n" for turn in conv_memory)
        prefix = "Based on the following client data:\n"
        suffix = f"\n\nAnswer the following query:\n{corrected_query}\nAnswer:"
        max_total_length = 8192
        max_new_tokens = 500
        max_input_tokens = max_total_length - max_new_tokens
        prefix_tokens = gpt2_tokenizer.encode(prefix, add_special_tokens=False)
        suffix_tokens = gpt2_tokenizer.encode(suffix, add_special_tokens=False)
        conv_tokens = gpt2_tokenizer.encode(conversation_context, add_special_tokens=False)
        available_tokens = max_input_tokens - (len(prefix_tokens) + len(suffix_tokens) + len(conv_tokens))
        context = build_context_from_docs(retrieved_docs, available_tokens)
        full_prompt = prefix + conversation_context + context + suffix

    final_token_count = len(gpt2_tokenizer.encode(full_prompt, add_special_tokens=False))
    logger.info("Final prompt token length: %d", final_token_count)
    logger.info("Constructed prompt (first 200 chars): %s", full_prompt[:200])
    
    answer = openai_inference(full_prompt, max_tokens=500)
    
    if user_id not in conversation_memory:
        conversation_memory[user_id] = []
    conversation_memory[user_id].append({"role": "user", "content": corrected_query})
    conversation_memory[user_id].append({"role": "assistant", "content": answer})
    query_cache[corrected_query] = answer
    return answer

def process_update_gsheet_natural_language(text):
    try:
        nl_text = text[len("update gsheet:"):].strip()
        prompt = (
            "You are an assistant that extracts update instructions from natural language text. "
            "Given the following text, output a valid JSON object containing update instructions with the following keys if mentioned: "
            "client_email, account_code, company_name, prospect_geography, prospect_title, prospect_industries, "
            "qualification_ranges, qualifiers, service_offered, offer_description, offer_usp, case_studies, "
            "do_not_reveal, awards, sales_assets, sales_hooks, do_not_contact_list, pabbly_placeholder, timestamp, "
            "top_challenges, tools_tried, leads_per_month, top_competitors, approach_better_than, direct_competitors, "
            "ideal_clients, price_range, status, offer_for_outreach, lead_sources, company_website, first_name, last_name, phone_number. "
            "The JSON must include either 'client_email' or 'account_code'. "
            "Here is the text:\n\n" + nl_text + "\n\nOutput only the JSON object."
        )
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=150,
        )
        json_output = response.choices[0].message.content.strip()
        data = json.loads(json_output)
        return data
    except Exception as e:
        logger.error("Error processing natural language update: %s", e)
        return None

def process_update_gsheet_command(text):
    if not text.lower().startswith("update gsheet:"):
        if text.lower().startswith("update ") and not text.lower().startswith("update memory:"):
            text = "update gsheet:" + text[len("update "):]
    update_content = text[len("update gsheet:"):].strip()
    try:
        data = json.loads(update_content)
    except Exception:
        data = process_update_gsheet_natural_language(text)
    if not data:
        return "Failed to parse update instructions."
    client_email = data.get("client_email")
    account_code = data.get("account_code")
    if not client_email and not account_code:
        return "Either client email or account code is required for updating the Google Sheet."
    updates = {k: v for k, v in data.items() if k not in ["client_email", "account_code"]}
    result = update_gsheet_row(client_email, account_code, updates)
    sync_client_data()
    return result

# -------------------------------
# Free-Text Memory Update Processing
# -------------------------------
def process_memory_update_free_text(text, user):
    try:
        memory_content = text[len("update memory:"):].strip()
        if not memory_content:
            return "No memory content provided."
        if user not in conversation_memory:
            conversation_memory[user] = []
        conversation_memory[user].append({"role": "system", "content": memory_content})
        logger.info("Updated memory for user %s with content: %s", user, memory_content)
        return "✅ Memory updated successfully."
    except Exception as e:
        logger.error("Error updating memory: %s", e)
        return f"Error updating memory: {e}"

# -------------------------------
# Slack Bot Setup Using Slack Bolt
# -------------------------------
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

app = App(token=SLACK_BOT_TOKEN)

@app.event("app_mention")
def handle_app_mention_events(body, say, logger):
    try:
        event = body.get("event", {})
        text = event.get("text", "")
        user = event.get("user")
        logger.info("Received app_mention from user %s: %s", user, text)
        cleaned_text = re.sub(r"<@[^>]+>", "", text).strip()
        if cleaned_text.lower().startswith("sync data:"):
            response_text = sync_client_data()
        elif cleaned_text.lower().startswith("update gsheet:") or (cleaned_text.lower().startswith("update ") and not cleaned_text.lower().startswith("update memory:")):
            response_text = process_update_gsheet_command(cleaned_text)
        elif cleaned_text.lower().startswith("update memory:"):
            response_text = process_memory_update_free_text(cleaned_text, user)
        else:
            response_text = process_query_offline(cleaned_text, user)
        say(response_text)
    except Exception as e:
        logger.error("Error handling app_mention event: %s", e)
        say("Sorry, an error occurred processing your request.")

@app.event("message")
def handle_direct_messages(body, say, logger):
    try:
        event = body.get("event", {})
        if event.get("subtype") is None:
            channel_type = event.get("channel_type")
            text = event.get("text", "")
            user = event.get("user")
            if channel_type == "im":
                logger.info("Received direct message from user %s: %s", user, text)
                cleaned_text = text.strip()
            else:
                cleaned_text = re.sub(r"<@[^>]+>", "", text).strip()
                logger.info("Cleaned channel message text: %s", cleaned_text)
            if cleaned_text.lower().startswith("sync data:"):
                response_text = sync_client_data()
            elif cleaned_text.lower().startswith("update gsheet:") or (cleaned_text.lower().startswith("update ") and not cleaned_text.lower().startswith("update memory:")):
                response_text = process_update_gsheet_command(cleaned_text)
            elif cleaned_text.lower().startswith("update memory:"):
                response_text = process_memory_update_free_text(cleaned_text, user)
            else:
                response_text = process_query_offline(cleaned_text, user)
            say(response_text)
    except Exception as e:
        logger.error("Error handling direct message event: %s", e)
        say("Sorry, an error occurred processing your message.")

# -------------------------------
# Main Entry Point: Initialize Data & Start Slack Bot
# -------------------------------
if __name__ == "__main__":
    initialize_client_data()
    logger.info("Starting Slack Bot with OpenAI-powered Retrieval-Augmented Generation in Socket Mode...")
    handler = SocketModeHandler(app, SLACK_APP_TOKEN)
    handler.start()
