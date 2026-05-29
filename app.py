import os
import re
import requests
from flask import Flask, request, jsonify, send_from_directory, make_response
from flask_cors import CORS
import pandas as pd
import traceback
import random

def load_env():
    if os.path.exists('.env'):
        with open('.env', 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    parts = line.split('=', 1)
                    k = parts[0].strip()
                    v = parts[1].strip().strip('"').strip("'")
                    os.environ[k] = v
load_env()

def get_file_map(counseling_type, round_selected):
    base_dir = os.path.join(counseling_type.upper(), round_selected.upper())
    file_map = {}
    if os.path.exists(base_dir):
        files = os.listdir(base_dir)
        for f in files:
            fname = f.lower()
            if 'gfti' in fname or fname == 'data.csv':
                file_map['GFTI'] = os.path.join(base_dir, f)
            elif 'nit' in fname:
                file_map['NIT'] = os.path.join(base_dir, f)
            elif 'iiit' in fname:
                file_map['IIIT'] = os.path.join(base_dir, f)
    return file_map

def get_category_variants(category_name):
    if not category_name:
        return []
    cat = category_name.upper().strip()
    if cat in ['OPEN', 'GENERAL', 'UR']:
        return ['OPEN']
    elif cat in ['OBC', 'OBC-NCL', 'OBC NCL']:
        return ['OBC-NCL']
    elif cat in ['EWS']:
        return ['EWS']
    elif cat in ['SC']:
        return ['SC']
    elif cat in ['ST']:
        return ['ST']
    elif 'OPEN' in cat and 'PWD' in cat:
        return ['OPEN (PWD)', 'OPEN (PwD)', 'OPEN PWD', 'OPEN PwD', 'OPEN(PWD)', 'OPEN(PwD)', 'PWD OPEN']
    elif 'OBC' in cat and 'PWD' in cat:
        return ['OBC-NCL (PWD)', 'OBC-NCL (PwD)', 'OBC-NCL PWD', 'OBC-NCL(PWD)', 'OBC-NCL(PwD)', 'PWD OBC-NCL']
    elif 'EWS' in cat and 'PWD' in cat:
        return ['EWS (PWD)', 'EWS (PwD)', 'EWS PWD', 'EWS(PWD)', 'EWS(PwD)', 'PWD EWS']
    elif 'SC' in cat and 'PWD' in cat:
        return ['SC (PWD)', 'SC (PwD)', 'SC PWD', 'SC(PWD)', 'SC(PwD)', 'PWD SC']
    elif 'ST' in cat and 'PWD' in cat:
        return ['ST (PWD)', 'ST (PwD)', 'ST PWD', 'ST(PWD)', 'ST(PwD)', 'PWD ST']
    return [category_name, category_name.upper(), category_name.lower()]

def safe_read_csv(filename):
    try:
        return pd.read_csv(filename, on_bad_lines='skip')
    except TypeError:
        return pd.read_csv(filename, error_bad_lines=False, warn_bad_lines=False)

def is_data_query(message):
    msg_lower = message.lower()
    
    # 1. Rank indicators: 4-6 digit numbers, lakh/thousand terms, or numbers accompanied by rank/crl
    if re.search(r'\b\d{4,6}\b', msg_lower) or re.search(r'\b\d+(?:\.\d+)?\s*(?:lakh|lakhs|l|k|thousand|thousands)\b', msg_lower):
        return True
    if re.search(r'\b(?:rank|crl|air)\s*(?:of|is|equals)?\s*#?\d{1,7}\b', msg_lower) or re.search(r'\b\d{1,7}\s*(?:rank|crl|air)\b', msg_lower):
        return True
        
    # 2. Database/prediction keywords
    data_keywords = [
        'cutoff', 'cut-off', 'closing rank', 'opening rank', 'closing', 'opening',
        'safe', 'low chance', 'chance', 'probability', 'unlock', 'seat',
        'nit', 'iiit', 'gfti', 'josaa', 'csab', 'quota', 'counseling', 'counselling',
        'crl', 'obc', 'ncl', 'ews', 'sc', 'st', 'pwd', 'category', 'gender',
        'neutral', 'female', 'male', 'open', 'general', 'state', 'home', 'other'
    ]
    if any(re.search(rf'\b{kw}\b', msg_lower) for kw in data_keywords):
        return True
        
    # 3. Branch keywords
    branch_keywords = ['cse', 'ece', 'eee', 'mech', 'civil', 'branch', 'computer science', 'information technology']
    if any(re.search(rf'\b{kw}\b', msg_lower) for kw in branch_keywords):
        return True
        
    # 4. Known college keywords
    college_keywords = ['jaipur', 'allahabad', 'bhopal', 'jalandhar', 'calicut', 'delhi', 'agartala', 'durgapur', 'goa', 'hamirpur', 'surathkal', 'trichy', 'warangal', 'patna', 'rourkela', 'silchar', 'srinagar', 'surat', 'kurukshetra', 'jamshedpur', 'nagpur', 'shibpur', 'puducherry']
    if any(re.search(rf'\b{kw}\b', msg_lower) for kw in college_keywords):
        return True
        
    return False

def get_gemini_response(prompt, history=None):
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return "GEMINI_API_KEY is not configured in the Flask backend environment. Please set your API key in the environment to chat with me!"
        
    # API endpoint for Gemini 2.5 Flash
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
    
    # System instruction (empathetic, concise, student-friendly counselor)
    system_instruction = (
        "You are CampusCipher AI, an empathetic, friendly, and supportive JEE college counseling assistant. "
        "Your target audience is stressed JEE engineering aspirants. Act natural, warm, and comforting. "
        "Keep your responses extremely natural, concise (under 2-3 sentences where possible), and emotionally aware. "
        "Use dynamic student-friendly chat language (like 'bro', 'bruh', 'chilling', 'take a breath'). "
        "Do NOT talk about specific cutoff numbers or predict seats in this mode, as those are handled by a separate local data system. "
        "Do NOT repeat the same robotic templates."
    )
    
    # Build payload contents incorporating chat history
    contents = []
    
    if history:
        # Prevent consecutive role violation if history already has the current prompt
        if len(history) > 0 and history[-1].get("content") == prompt:
            history = history[:-1]
            
        for msg in history:
            role = "user" if msg.get("role") == "user" else "model"
            contents.append({
                "role": role,
                "parts": [{"text": msg.get("content", "")}]
            })
            
    # Append current user prompt
    contents.append({
        "role": "user",
        "parts": [{"text": prompt}]
    })
    
    payload = {
        "contents": contents,
        "systemInstruction": {
            "parts": [{"text": system_instruction}]
        },
        "generationConfig": {
            "temperature": 0.7,
            "maxOutputTokens": 300
        }
    }
    
    try:
        headers = {"Content-Type": "application/json"}
        response = requests.post(url, json=payload, headers=headers, timeout=10)
        if response.status_code == 200:
            res_data = response.json()
            text = res_data['candidates'][0]['content']['parts'][0]['text']
            return text.strip()
        else:
            print(f"Gemini API error status {response.status_code}: {response.text}")
            return "Ah, I encountered a connection hiccup with my AI brain. Let's try again in a moment!"
    except Exception as e:
        print(f"Exception during Gemini call: {e}")
        return "I had trouble connecting to the AI server. Make sure your internet is active!"


app = Flask(__name__)
CORS(app)  # Enable CORS for frontend requests

# PWA Root Static Asset Serving
@app.route('/')
def serve_index():
    return send_from_directory('.', 'index.html')

@app.route('/manifest.json')
def serve_manifest():
    return send_from_directory('.', 'manifest.json')

@app.route('/service-worker.js')
def serve_service_worker():
    response = make_response(send_from_directory('.', 'service-worker.js'))
    response.headers['Content-Type'] = 'application/javascript'
    response.headers['Service-Worker-Allowed'] = '/'
    return response

@app.route('/icon-192.png')
def serve_icon192():
    return send_from_directory('.', 'icon-192.png')

@app.route('/icon-512.png')
def serve_icon512():
    return send_from_directory('.', 'icon-512.png')

@app.route('/favicon.png')
def serve_favicon():
    return send_from_directory('.', 'favicon.png')

USER_CONTEXT = {
    "last_rank": None,
    "last_category": None,
    "last_gender": None,
    "last_quota": None,
    "last_college": None,
    "last_branch": None,
    "last_counseling_type": None,
    "last_round": None,
    "last_institute_type": None,
    "last_topic": None
}

CHAT_CONTEXT = {
    "last_rank": None,
    "last_category": None,
    "last_gender": None,
    "last_quota": None,
    "last_college": None,
    "last_branch": None,
    "last_counseling_type": None,
    "last_round": None,
    "last_institute_type": None,
    "last_topic": None
}

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid request, JSON required"}), 400

        institute_type = data.get('institute_type')
        rank = data.get('rank')
        quota = data.get('quota')
        category = data.get('category')
        gender = data.get('gender')
        college = data.get('college')
        branch = data.get('branch')

        # Update USER_CONTEXT
        global USER_CONTEXT
        try:
            if rank:
                USER_CONTEXT['last_rank'] = int(rank)
            if category:
                USER_CONTEXT['last_category'] = category
            if gender:
                USER_CONTEXT['last_gender'] = gender
            if quota:
                USER_CONTEXT['last_quota'] = quota
            if college:
                USER_CONTEXT['last_college'] = college
            if branch:
                USER_CONTEXT['last_branch'] = branch
            USER_CONTEXT['last_topic'] = 'prediction'
        except Exception:
            pass

        round_selected = data.get('round', 'Round 3')
        counseling_type = data.get('counseling_type', 'CSAB')
        # Map institute to filename
        file_map = get_file_map(counseling_type, round_selected)

        df = None
        if institute_type == 'All':
            dfs = []
            for itype, fname in file_map.items():
                try:
                    temp_df = safe_read_csv(fname)
                    temp_df['Institute_Type'] = itype
                    dfs.append(temp_df)
                except Exception as e:
                    print(f"Error loading {fname}: {e}")
            if dfs:
                df = pd.concat(dfs, ignore_index=True)
        else:
            filename = file_map.get(institute_type)
            if not filename:
                return jsonify({"error": "Invalid or missing institute type."}), 400

            try:
                df = safe_read_csv(filename)
                df['Institute_Type'] = institute_type
            except Exception as e:
                print(f"Error loading {filename}: {e}")

        if df is None:
            return jsonify({"error": "Failed to load dataset."}), 500

        # Validate input
        if rank is None or not institute_type or not quota or not category or not gender:
            return jsonify({"error": "Missing required fields."}), 400

        def get_buffer(r):
            if r <= 10000: return 1000
            elif r <= 20000: return 2500
            elif r <= 50000: return 4500
            elif r <= 100000: return 7500
            elif r <= 200000: return 11000
            elif r <= 500000: return 18000
            elif r <= 1000000: return 25000
            else: return 30000


        try:
            rank = int(rank)
        except ValueError:
            return jsonify({"error": "Rank must be a valid number."}), 400

        if gender == 'Female-only':
            gender_condition = df['Gender'].isin(['Female-only', 'Gender-Neutral'])
        else:
            gender_condition = df['Gender'] == gender

        if quota == "All India + Other State":
            quota_condition = df['Quota'].isin(["All India", "Other State"])
        else:
            quota_condition = df['Quota'] == quota

        # Apply logic: Filter rows based on rank <= Closing Rank + buffer
        rank_buffer = get_buffer(rank)
        cat_variants = get_category_variants(category)
        filtered_df = df[
            quota_condition &
            (df['Category'].isin(cat_variants)) &
            gender_condition &
            (df['Closing Rank'] >= (rank - rank_buffer))
        ].copy()

        filtered_df['Status'] = filtered_df['Closing Rank'].apply(lambda x: 'Safe' if x >= rank else 'Low Chance')

        if college:
            filtered_df = filtered_df[filtered_df['College'].str.contains(college, case=False, na=False)]

        if branch:
            filtered_df = filtered_df[filtered_df['Branch'].str.contains(branch, case=False, na=False)]

        if institute_type == 'All':
            type_order = {'NIT': 1, 'IIIT': 2, 'GFTI': 3}
            filtered_df['Type_Order'] = filtered_df['Institute_Type'].map(type_order)
            filtered_df = filtered_df.sort_values(by=['Type_Order', 'Closing Rank'])
        else:
            filtered_df = filtered_df.sort_values(by=['Closing Rank'])

        # Prepare response list of objects
        results = filtered_df[['College', 'Branch', 'Opening Rank', 'Closing Rank', 'Institute_Type', 'Quota', 'Status']].to_dict(orient='records')
        
        return jsonify(results), 200

    except Exception as e:
        print(f"Error occurring in /predict endpoint: {e}")
        traceback.print_exc()
        return jsonify({"error": "An internal server error occurred processing the prediction."}), 500

@app.route('/search', methods=['POST'])
def search_college():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid request, JSON required"}), 400

        college_name = data.get('college_name', '').strip()
        branch = data.get('branch', '').strip()
        category = data.get('category', '').strip()
        gender = data.get('gender', '').strip()

        if not college_name:
            return jsonify({"error": "College name is required for search."}), 400

        round_selected = data.get('round', 'Round 3')
        counseling_type = data.get('counseling_type', 'CSAB')
        file_map = get_file_map(counseling_type, round_selected)

        dfs = []
        for itype, fname in file_map.items():
            try:
                temp_df = safe_read_csv(fname)
                temp_df['Institute_Type'] = itype
                dfs.append(temp_df)
            except Exception as e:
                print(f"Error loading {fname}: {e}")
        
        if not dfs:
            return jsonify({"error": "Failed to load datasets."}), 500
            
        df = pd.concat(dfs, ignore_index=True)

        filtered_df = df[df['College'].str.contains(college_name, case=False, na=False)].copy()

        if branch:
            filtered_df = filtered_df[filtered_df['Branch'].str.contains(branch, case=False, na=False)]
            
        if category:
            cat_variants = get_category_variants(category)
            cat_variants_lower = [c.lower() for c in cat_variants]
            filtered_df = filtered_df[filtered_df['Category'].str.lower().isin(cat_variants_lower)]

        if gender:
            filtered_df = filtered_df[filtered_df['Gender'].str.lower() == gender.lower()]

        filtered_df = filtered_df.sort_values(by=['College', 'Closing Rank'])

        results = filtered_df[['College', 'Branch', 'Category', 'Quota', 'Gender', 'Closing Rank', 'Institute_Type']].to_dict(orient='records')
        
        return jsonify(results), 200

    except Exception as e:
        print(f"Error occurring in /search endpoint: {e}")
        traceback.print_exc()
        return jsonify({"error": "An internal server error occurred processing the search."}), 500

@app.route('/lowest_cutoff', methods=['POST'])
def lowest_cutoff():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid request, JSON required"}), 400

        institute_type = data.get('institute_type', '').strip()
        branch = data.get('branch', '').strip()
        category = data.get('category', '').strip()
        quota = data.get('quota', '').strip()
        gender = data.get('gender', '').strip()

        if not institute_type:
            return jsonify({"error": "Institute type is required."}), 400

        round_selected = data.get('round', 'Round 3')
        counseling_type = data.get('counseling_type', 'CSAB')
        file_map = get_file_map(counseling_type, round_selected)

        dfs = []
        if institute_type == 'All':
            for itype, fname in file_map.items():
                try:
                    temp_df = safe_read_csv(fname)
                    temp_df['Institute_Type'] = itype
                    dfs.append(temp_df)
                except Exception as e:
                    print(f"Error loading {fname}: {e}")
        else:
            fname = file_map.get(institute_type)
            if not fname:
                return jsonify({"error": "Invalid institute type."}), 400
            try:
                temp_df = safe_read_csv(fname)
                temp_df['Institute_Type'] = institute_type
                dfs.append(temp_df)
            except Exception as e:
                print(f"Error loading {fname}: {e}")

        if not dfs:
            return jsonify({"error": "Failed to load datasets."}), 500
            
        df = pd.concat(dfs, ignore_index=True)

        if branch:
            df = df[df['Branch'].str.contains(branch, case=False, na=False)]
            
        if category:
            cat_variants = get_category_variants(category)
            cat_variants_lower = [c.lower() for c in cat_variants]
            df = df[df['Category'].str.lower().isin(cat_variants_lower)]
            
        if quota:
            if quota.lower() == "all india + other state":
                df = df[df['Quota'].str.lower().isin(['all india', 'other state'])]
            else:
                df = df[df['Quota'].str.lower() == quota.lower()]
            
        if gender:
            df = df[df['Gender'].str.lower() == gender.lower()]

        if df.empty:
            return jsonify([]), 200

        # Find the row with the maximum Closing Rank
        max_idx = df['Closing Rank'].idxmax()
        lowest_cutoff_row = df.loc[max_idx]

        results = [{
            'College': lowest_cutoff_row['College'],
            'Branch': lowest_cutoff_row['Branch'],
            'Category': lowest_cutoff_row['Category'],
            'Quota': lowest_cutoff_row['Quota'],
            'Gender': lowest_cutoff_row['Gender'],
            'Opening Rank': int(lowest_cutoff_row['Opening Rank']),
            'Closing Rank': int(lowest_cutoff_row['Closing Rank']),
            'Institute_Type': lowest_cutoff_row['Institute_Type']
        }]
        
        return jsonify(results), 200

    except Exception as e:
        print(f"Error occurring in /lowest_cutoff endpoint: {e}")
        traceback.print_exc()
        return jsonify({"error": "An internal server error occurred processing the search."}), 500

@app.route('/api/colleges', methods=['GET'])
def get_colleges():
    try:
        file_map = get_file_map('CSAB', 'Round 3')
        dfs = []
        for itype, fname in file_map.items():
            try:
                temp_df = safe_read_csv(fname)
                dfs.append(temp_df)
            except Exception as e:
                pass
        if not dfs:
            return jsonify([]), 200
            
        df = pd.concat(dfs, ignore_index=True)
        colleges = df['College'].dropna().unique().tolist()
        colleges.sort()
        
        return jsonify(colleges), 200
    except Exception as e:
        print(f"Error fetching colleges: {e}")
        return jsonify([]), 500

import re

def parse_chat_message(message):
    message_lower = message.lower()
    
    # 1. Rank Extraction
    rank = None
    # Check for "X lakh" or "X.Y lakh"
    lakh_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:lakh|lakhs|l)\b', message_lower)
    if lakh_match:
        try:
            rank = int(float(lakh_match.group(1)) * 100000)
        except ValueError:
            pass
            
    # Check for "X k" or "X thousand"
    if not rank:
        k_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:k|thousand|thousands)\b', message_lower)
        if k_match:
            try:
                rank = int(float(k_match.group(1)) * 1000)
            except ValueError:
                pass
                
    # Check for rank/crl keyword + short/long numbers (e.g. rank 500 or 500 rank)
    if not rank:
        rank_pattern_match = re.search(r'\b(?:rank|crl|air)\s*(?:of|is|equals)?\s*#?(\d{1,7})\b', message_lower)
        if rank_pattern_match:
            try:
                rank = int(rank_pattern_match.group(1))
            except ValueError:
                pass
    if not rank:
        rank_pattern_match = re.search(r'\b(\d{1,7})\s*(?:rank|crl|air)\b', message_lower)
        if rank_pattern_match:
            try:
                rank = int(rank_pattern_match.group(1))
            except ValueError:
                pass

    # Check for raw numbers (4-6 digits)
    if not rank:
        num_match = re.search(r'\b(\d{4,6})\b', message_lower)
        if num_match:
            try:
                rank = int(num_match.group(1))
            except ValueError:
                pass
                
    # 2. Category Extraction
    category = None
    is_pwd = 'pwd' in message_lower or 'physical' in message_lower or 'handicap' in message_lower or 'disabled' in message_lower
    if re.search(r'\bobc\b|\bncl\b', message_lower):
        category = "OBC-NCL (PwD)" if is_pwd else "OBC-NCL"
    elif re.search(r'\bsc\b', message_lower):
        category = "SC (PwD)" if is_pwd else "SC"
    elif re.search(r'\bst\b', message_lower):
        category = "ST (PwD)" if is_pwd else "ST"
    elif re.search(r'\bews\b', message_lower):
        category = "EWS (PwD)" if is_pwd else "EWS"
    elif re.search(r'\bgeneral\b|\bopen\b|\bur\b', message_lower):
        category = "OPEN (PwD)" if is_pwd else "OPEN"
    elif is_pwd:
        category = "OPEN (PwD)"
        
    # 3. Quota Extraction
    quota = None
    if "home state" in message_lower or re.search(r'\bhs\b', message_lower):
        quota = "Home State"
    elif "other state" in message_lower or re.search(r'\bos\b', message_lower):
        quota = "Other State"
    elif "all india" in message_lower or re.search(r'\bai\b', message_lower):
        quota = "All India"
        
    # 4. Gender Extraction
    gender = None
    if re.search(r'\bfemale\b|\bgirl\b|\bwoman\b|\bgirls\b', message_lower):
        gender = "Female-only"
    elif re.search(r'\bmale\b|\bboy\b|\bboys\b|\bneutral\b', message_lower):
        gender = "Gender-Neutral"
        
    # 5. College Extraction
    college_keywords = {
        "jaipur": "Jaipur",
        "allahabad": "Allahabad",
        "bhopal": "Bhopal",
        "jalandhar": "Jalandhar",
        "calicut": "Calicut",
        "delhi": "Delhi",
        "agartala": "Agartala",
        "durgapur": "Durgapur",
        "goa": "Goa",
        "hamirpur": "Hamirpur",
        "surathkal": "Surathkal",
        "trichy": "Tiruchirappalli",
        "tiruchirappalli": "Tiruchirappalli",
        "warangal": "Warangal",
        "patna": "Patna",
        "raipur": "Raipur",
        "rourkela": "Rourkela",
        "silchar": "Silchar",
        "srinagar": "Srinagar",
        "surat": "Surat",
        "kurukshetra": "Kurukshetra",
        "jamshedpur": "Jamshedpur",
        "nagpur": "Nagpur",
        "shibpur": "Shibpur",
        "puducherry": "Puducherry",
        "uttarakhand": "Uttarakhand",
        "mizoram": "Mizoram",
        "nagaland": "Nagaland",
        "manipur": "Manipur",
        "meghalaya": "Meghalaya",
        "sikkim": "Sikkim",
        "arunachal": "Arunachal",
        "andhra": "Andhra"
    }
    
    extracted_college = None
    for keyword, col_name in college_keywords.items():
        if re.search(rf'\b{keyword}\b', message_lower):
            extracted_college = col_name
            break
            
    # 6. Branch Extraction
    branch_keywords = {
        "cse": "Computer Science",
        "computer science": "Computer Science",
        "it": "Information Technology",
        "information tech": "Information Technology",
        "ece": "Electronics",
        "electronics": "Electronics",
        "vlsi": "VLSI",
        "mech": "Mechanical",
        "mechanical": "Mechanical",
        "civil": "Civil",
        "chem": "Chemical",
        "chemical": "Chemical",
        "biotech": "Bio Technology",
        "bio technology": "Bio Technology",
        "electrical": "Electrical",
        "eee": "Electrical and Electronics",
        "metallurgy": "Metallurgical",
        "materials": "Materials",
        "production": "Production",
        "industrial": "Industrial",
        "ai": "Artificial Intelligence",
        "data science": "Data Science"
    }
    
    # 7. Counseling Type Extraction
    counseling_type = None
    if re.search(r'\bjosaa\b|\bjosa\b', message_lower):
        counseling_type = "JOSAA"
    elif re.search(r'\bcsab\b', message_lower):
        counseling_type = "CSAB"
        
    # 8. Round Extraction
    round_val = None
    round_match = re.search(r'\b(?:round|r)\s*(\d)\b', message_lower)
    if round_match:
        round_num = round_match.group(1)
        if round_num in ['1', '2', '3', '4', '6']:
            round_val = f"Round {round_num}"

    # 9. Institute Type Extraction
    institute_type = None
    if re.search(r'\bnit\b|\bnits\b', message_lower):
        institute_type = "NIT"
    elif re.search(r'\biiit\b|\biiits\b', message_lower):
        institute_type = "IIIT"
    elif re.search(r'\bgfti\b|\bgftis\b', message_lower):
        institute_type = "GFTI"
    elif "all" in message_lower or "any" in message_lower:
        institute_type = "All"

    extracted_branch = None
    for keyword, br_name in branch_keywords.items():
        if re.search(rf'\b{keyword}\b', message_lower):
            extracted_branch = br_name
            break
            
    return {
        "rank": rank,
        "category": category,
        "quota": quota,
        "gender": gender,
        "college": extracted_college,
        "branch": extracted_branch,
        "counseling_type": counseling_type,
        "round": round_val,
        "institute_type": institute_type
    }

def load_all_cutoff_data():
    dfs = []
    
    # 1. Look for main nit_data.csv in current directory
    if os.path.exists('nit_data.csv'):
        try:
            df = safe_read_csv('nit_data.csv')
            df['Counseling'] = 'CSAB'
            df['Round'] = 'Round 3'
            df['Institute_Type'] = 'NIT'
            dfs.append(df)
        except Exception:
            pass

    # 2. Automatically discover files in CSAB and JOSAA folders
    for counseling in ['CSAB', 'JOSAA']:
        if os.path.exists(counseling):
            rounds = os.listdir(counseling)
            for r in rounds:
                round_dir = os.path.join(counseling, r)
                if os.path.isdir(round_dir):
                    files = os.listdir(round_dir)
                    for f in files:
                        if f.lower().endswith('.csv'):
                            path = os.path.join(round_dir, f)
                            try:
                                df = safe_read_csv(path)
                                df['Counseling'] = counseling
                                df['Round'] = r
                                if 'gfti' in f.lower() or f.lower() == 'data.csv':
                                    df['Institute_Type'] = 'GFTI'
                                elif 'nit' in f.lower():
                                    df['Institute_Type'] = 'NIT'
                                elif 'iiit' in f.lower():
                                    df['Institute_Type'] = 'IIIT'
                                else:
                                    df['Institute_Type'] = 'Other'
                                dfs.append(df)
                            except Exception:
                                pass
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return None

# Load dataset on startup
CHAT_DF = load_all_cutoff_data()

def rebuild_chat_context(history, current_message):
    context = {
        "last_rank": None,
        "last_category": None,
        "last_gender": None,
        "last_quota": None,
        "last_college": None,
        "last_branch": None,
        "last_counseling_type": None,
        "last_round": None,
        "last_institute_type": None
    }
    
    # We collect all user messages chronologically
    user_messages = []
    if history:
        for msg in history:
            if msg.get("role") == "user":
                user_messages.append(msg.get("content", ""))
                
    user_messages.append(current_message)
    
    # Process each user message in chronological order
    for msg in user_messages:
        parsed = parse_chat_message(msg)
        rank = parsed.get("rank")
        category = parsed.get("category")
        gender = parsed.get("gender")
        quota = parsed.get("quota")
        college = parsed.get("college")
        branch = parsed.get("branch")
        counseling_type = parsed.get("counseling_type")
        round_val = parsed.get("round")
        institute_type = parsed.get("institute_type")
        
        if rank is not None:
            # A new rank is specified! Reset context parameters not in this message
            context["last_rank"] = rank
            context["last_category"] = category
            context["last_gender"] = gender
            context["last_quota"] = quota
            context["last_college"] = college
            context["last_branch"] = branch
            context["last_counseling_type"] = counseling_type
            context["last_round"] = round_val
            context["last_institute_type"] = institute_type
        else:
            # No rank specified in this message, update only the fields that are present
            if category is not None: context["last_category"] = category
            if gender is not None: context["last_gender"] = gender
            if quota is not None: context["last_quota"] = quota
            if college is not None: context["last_college"] = college
            if branch is not None: context["last_branch"] = branch
            if counseling_type is not None: context["last_counseling_type"] = counseling_type
            if round_val is not None: context["last_round"] = round_val
            if institute_type is not None: context["last_institute_type"] = institute_type
            
    return context

def get_casual_response(message, context=None):
    msg_lower = message.lower()
    if context is None:
        context = {}
    
    rank = context.get("last_rank")
    category = context.get("last_category")
    college = context.get("last_college")
    branch = context.get("last_branch")
    
    # 1. "cooked" / exam failure / messed up
    if re.search(r'\b(cooked|messed\s*up|fail|failed|ruined|screwed|over|it\s*is\s*over)\b', msg_lower):
        responses = [
            "Depends... cooked beyond repair or just overthinking after seeing the opening/closing ranks? Let's check. What's the damage (rank)?",
            "Honestly, half the JEE candidates think they are cooked right now. Take a deep breath. Unless you've got a 10 lakh rank and only want NIT Trichy CSE, we can find a way. What rank did you get?",
            "Bruh, the counseling hasn't even started properly and you're calling yourself cooked? 😭 Tell me your rank, let's see how much we can save."
        ]
        return random.choice(responses)
        
    # 2. small chat slangs: bruh, bro, buddy
    if re.search(r'\b(bruh|bro|brother|buddy|dude|mate|yaar)\b', msg_lower):
        responses = [
            "Bruh. 💀 What did you see? Did the CSAB cutoffs scare you?",
            "Yes, bro? Tell me what's on your mind. Counseling stress is real.",
            "Bro, don't worry. I'm here. What's the JEE rank situation?"
        ]
        return random.choice(responses)
        
    # 3. laughing: lol, lmao, rofl, haha, xd
    if re.search(r'\b(lol|lmao|rofl|haha|hahaha|xd|lmfao)\b', msg_lower):
        responses = [
            "😭 Counseling season does this to everyone. One minute you're laughing, next minute you're analyzing closing ranks at 2 AM.",
            "Glad you're finding some humor in this chaos! But seriously, how's the counseling stress level?",
            "Haha, keeping a light mood is key. What are we planning next?"
        ]
        return random.choice(responses)
        
    # 4. thinking / hesitation: hmm, ok, fine, oh, i see
    if re.search(r'\b(hmm|hmmm|thinking|pondering)\b', msg_lower):
        responses = [
            "Hmm... deep thinking? Or just shocked by how fast the cutoffs closed last year?",
            "Thinking about backup options? That's smart. Share your rank, we can list some solid safeties.",
            "Hmm indeed. The JoSAA database has some wild cutoff jumps sometimes. What's on your mind?"
        ]
        return random.choice(responses)
        
    if re.search(r'\b(oh|ohh|ic|i\s*see|ah|ahh)\b', msg_lower):
        responses = [
            "Yeah... counseling choices are a lot to process. What did you think of the options?",
            "Oh? Did something catch your eye, or did a cutoff rank surprise you?",
            "Yep. It is what it is. But we still have options. Let's see what we should do next."
        ]
        return random.choice(responses)
        
    if re.search(r'\b(okay|ok|fine|sure|got\s*it|alright|kk)\b', msg_lower):
        responses = [
            "Cool. Ready to check some more options? Tell me a college or branch you're curious about.",
            "Alright! Let's keep moving. What rank or college are we searching next?",
            "Got it. Counseling is a step-by-step game. Let me know if you want to look up another cutoff."
        ]
        return random.choice(responses)
        
    # 5. Stress / Anxiety / Emotion
    if re.search(r'\b(stressed|anxious|stress|worried|worry|scared|confused|overwhelmed|panic|sad|crying|depressed|fear|tense|nervous|afraid)\b', msg_lower):
        if rank:
            responses = [
                f"Look, you have a rank of **{rank:,}** ({category}). Honestly, it is a decent rank! Yes, some high-tier CS seats might be tough, but you can comfortably lock several solid branches. Don't let the anxiety make you panic-freeze. Should we look at similar NIT options?",
                f"Anxiety is high, I get it. But with your rank of **{rank:,}**, you are definitely NOT out of the game. Let's look at the numbers: there are actual safe seats for you. What branch is your absolute priority?"
            ]
        else:
            responses = [
                "It's completely normal to feel this way. Counseling season feels like a giant high-stakes puzzle. Let's break it down together. What rank and category are we working with?",
                "I get it, the fear of making a wrong choice is real. But panic doesn't help. Let's look at actual cutoff statistics. Share your rank, and we can find some solid safety options to take the pressure off."
            ]
        return random.choice(responses)
        
    # 6. Gratitude / Thanks
    if re.search(r'\b(thank\s*you|thanks|thank\s*u|ty|awesome|great|perfect|wonderful|helpful|thx)\b', msg_lower):
        responses = [
            "You're very welcome! 😊 I'm glad I could help. Whenever you're ready, we can check more college options or branch details for your rank.",
            "Anytime, buddy! 🎓 Counseling is tricky, glad I could make it a bit clearer. Let me know if you have another question.",
            "Happy to help! Let me know if you want to check other branches or colleges next."
        ]
        return random.choice(responses)
        
    # 7. What do you think / Recommendations
    if re.search(r'\b(what\s*do\s*you\s*think|suggest|recommend|what\s*should\s*i\s*do|opinion|advice|any\s*ideas)\b', msg_lower):
        if rank:
            responses = [
                f"With your rank of **{rank:,}** ({category}), I think you should keep a balanced choice filling list. Put some ambitious NIT/IIIT options at the top, but definitely secure a few solid, high-probability GFTI/NIT seats in the middle. Should we explore some safeties?",
                f"Honestly, at **{rank:,}**, you have a decent shot at NITs if you're open to branches like Mechanical, Civil, or Metallurgy, or IIITs if you want ECE. My recommendation: don't chase CSE blindly in low-tier colleges if you can get a better college. What do you prefer?"
            ]
        else:
            responses = [
                "Hard to say without details! Give me your JEE Main Rank and category, and I'll give you a realistic recommendation based on the data.",
                "I need numbers to suggest anything solid, bro! 💀 Share your rank and category, let's see what's actually possible."
            ]
        return random.choice(responses)
        
    # 8. Creator / Identity
    if re.search(r'\b(who\s*made\s*you|who\s*created\s*you|your\s*creator|your\s*name|who\s*are\s*you|are\s*you\s*chatgpt|are\s*you\s*gpt)\b', msg_lower):
        responses = [
            "I am CampusCipher AI, your dedicated offline JEE College Counseling Assistant! 🎓 I was built to parse the official JoSAA and CSAB cutoff databases and provide exact, reliable answers without any hallucinations. How can I help you today?",
            "I'm CampusCipher AI, an offline counselor assistant designed to keep your search 100% data-accurate and zero-hallucination. Think of me as your personal cutoff guide!"
        ]
        return random.choice(responses)
        
    # 9. Capabilities
    if re.search(r'\b(what\s*can\s*you\s*do|capabilities|how\s*to\s*use|help\s*me|features|commands|what\s*do\s*you\s*do)\b', msg_lower):
        return ("I can act as your smart counseling guide! Here's what we can do:\n"
                "* **Analyze your rank:** Tell me your rank, category, and gender to see safe and borderline NIT/IIIT/GFTI options.\n"
                "* **Check cutoffs:** Ask about a specific college's closing ranks (e.g. *'NIT Jaipur cutoffs'*).\n"
                "* **Filter by branch:** Ask about specific fields (e.g. *'Where can I get CSE at 30k rank?'*).\n"
                "Just type your query and I will search the official datasets for you!")
                
    # 10. Greetings
    if re.search(r'\b(hello|hi|hey|greetings|good\s*morning|good\s*afternoon|good\s*evening|yoo|yo|heyy|heyyy)\b', msg_lower):
        responses = [
            "Hey! 🎓 How's the JEE counseling prep going? Stressed or chilling?",
            "Hello! Ready to mine some cutoffs or just here to rant about JEE?",
            "Hey there! Counseling Assistant on duty. What's on your mind today?"
        ]
        return random.choice(responses)
        
    # 11. Good night / goodbye
    if re.search(r'\b(good\s*night|bye|goodbye|see\s*you|gn|see\s*ya|exit|quit)\b', msg_lower):
        responses = [
            "Good night! 🌌 Go sleep and stop scrolling through opening/closing ranks at this hour. We can check more tomorrow.",
            "GN! Rest up. Counseling decisions are 100% better made when you're not sleep-deprived. Catch you later!",
            "Good night! Sleep well, buddy. I'll be here whenever you're ready to search again."
        ]
        return random.choice(responses)
        
    # 12. General "How are you"
    if re.search(r'\b(how\s*are\s*you|how\s*s\s*it\s*going|how\s*do\s*you\s*do|are\s*you\s*fine|how\s*r\s*u)\b', msg_lower):
        responses = [
            "I'm doing great, fully compiled and ready to search the CSAB/JoSAA databases! 🚀 How is your counseling prep going today?",
            "All systems normal, ready to parse some cutoffs! How about you? Doing okay under all this counseling stress?"
        ]
        return random.choice(responses)
        
    return None

@app.route('/api/chat', methods=['POST'])
def chat():
    global CHAT_DF
    try:
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({"response": "I didn't receive any message. How can I help you today?"}), 400
            
        message = data.get('message', '').strip()
        if not message:
            return jsonify({"response": "Please type a message to start counseling."}), 400
            
        history = data.get('history', [])
            
        # Reconstruct active conversation profile parameters completely stateless from the chat history
        session_context = rebuild_chat_context(history, message)
            
        # DUAL-MODE ROUTER:
        if not is_data_query(message):
            # 1. Gemini API (Online conversational mode)
            api_key = os.environ.get("GEMINI_API_KEY")
            if api_key:
                gemini_reply = get_gemini_response(message, history)
                if "hiccup" not in gemini_reply and "trouble connecting" not in gemini_reply and "not configured" not in gemini_reply:
                    return jsonify({"response": gemini_reply}), 200
                
            # 2. Local fallback (Offline conversational mode)
            casual_response = get_casual_response(message, session_context)
            if casual_response:
                return jsonify({"response": casual_response}), 200
                
            # 3. Default fallback
            prev_rank = session_context.get('last_rank')
            prev_category = session_context.get('last_category')
            if prev_rank:
                fallback_msg = (f"I'm here as your smart JEE Counseling Assistant. 🎓 "
                                f"Earlier we were looking at options for your rank of **{prev_rank:,}**" + (f" ({prev_category})" if prev_category else "") + ". "
                                f"We can continue checking more branches or specific college cutoffs for that rank, "
                                f"or test a new rank if you prefer! What details should we search next?")
            else:
                fallback_msg = ("I am your dedicated JEE Counseling Assistant. 🎓 To search the official "
                                "JoSAA/CSAB cutoff databases and predict your best options, please share your "
                                "**JEE Main Rank**, **Category**, and **Gender** (e.g., *'Can I get NIT Jaipur at 70k rank OPEN?'*).")
            return jsonify({"response": fallback_msg}), 200

        # --- PREDICTION/DATA MODE ---
        if CHAT_DF is None:
            CHAT_DF = load_all_cutoff_data()
            
        if CHAT_DF is None:
            return jsonify({"response": "This information is not available in the current dataset."}), 200

        current_rank = session_context.get('last_rank')
        current_category = session_context.get('last_category')
        current_quota = session_context.get('last_quota')
        current_gender = session_context.get('last_gender')
        current_counseling_type = session_context.get('last_counseling_type')
        current_round = session_context.get('last_round')
        current_institute_type = session_context.get('last_institute_type')
        
        # Resolve 'final' or 'last' round if counseling type is known
        if "final" in message.lower() or "last" in message.lower():
            if not current_round:
                if current_counseling_type == "CSAB":
                    current_round = "Round 3"
                else:
                    current_round = "Round 6"

        # Check for missing mandatory fields
        missing = []
        if not current_rank:
            missing.append("JEE Main Rank")
        if not current_category:
            missing.append("Category (e.g. OPEN, OBC-NCL, EWS, SC, ST or PwD)")
        if not current_quota:
            missing.append("Quota (All India, Home State, or Other State)")
        if not current_gender:
            missing.append("Gender Profile (Gender-Neutral or Female-only)")
        if not current_counseling_type:
            missing.append("Counseling Type (JoSAA or CSAB)")
        if not current_round:
            missing.append("Round (e.g. Round 1, Round 2, Round 3, Round 4, Round 6)")
            
        if missing:
            bullet_points = "\n".join([f"* **{item}**" for item in missing])
            inst_hint = ""
            if not current_institute_type:
                inst_hint = "\n* **Preferred Institute Type** (NIT, IIIT, GFTI, or all - *optional*)"
                
            response_msg = (
                "Sure — I can help you find your best engineering college options! "
                "However, to query the official cutoff database accurately, please tell me your:\n\n"
                f"{bullet_points}{inst_hint}\n\n"
                "Just tell me these parameters (e.g., *'OBC category, other state, gender neutral, CSAB Round 3'*) and I will predict your matches instantly!"
            )
            return jsonify({"response": response_msg}), 200

        # Re-assign validated consolidated values for prediction engine
        category = current_category
        gender = current_gender
        quota = current_quota
        rank = current_rank
        counseling_type = current_counseling_type
        round_val = current_round
        inst_type = current_institute_type if (current_institute_type and current_institute_type.lower() != 'all') else 'All'
        college = session_context.get('last_college')
        branch = session_context.get('last_branch')
        
        # Filter matching rows
        df_filtered = CHAT_DF.copy()
        
        # Filter Counseling Type
        if counseling_type:
            df_filtered = df_filtered[df_filtered['Counseling'].str.upper() == counseling_type.upper()]
            
        # Filter Round
        if round_val:
            df_filtered = df_filtered[df_filtered['Round'].str.upper() == round_val.upper()]
            
        # Filter Institute Type
        if inst_type != 'All':
            df_filtered = df_filtered[df_filtered['Institute_Type'].str.upper() == inst_type.upper()]
            
        # Filter Category (Required matching)
        if category:
            cat_variants = get_category_variants(category)
            df_filtered = df_filtered[df_filtered['Category'].isin(cat_variants)]
            
        # Filter Gender
        if gender == 'Female-only':
            df_filtered = df_filtered[df_filtered['Gender'].isin(['Female-only', 'Gender-Neutral'])]
        else:
            df_filtered = df_filtered[df_filtered['Gender'] == 'Gender-Neutral']
            
        # Filter Quota
        if quota:
            if quota == "All India + Other State":
                df_filtered = df_filtered[df_filtered['Quota'].isin(["All India", "Other State"])]
            else:
                df_filtered = df_filtered[df_filtered['Quota'] == quota]
            
        # Filter College (If mentioned)
        if college:
            df_filtered = df_filtered[df_filtered['College'].str.contains(college, case=False, na=False)]
            
        # Filter Branch (If mentioned)
        if branch:
            df_filtered = df_filtered[df_filtered['Branch'].str.contains(branch, case=False, na=False)]
            
        if df_filtered.empty:
            return jsonify({"response": "This information is not available in the current dataset."}), 200

        # Generate Answer
        df_filtered = df_filtered.sort_values(by='Closing Rank')
        
        # Safe: rank <= Closing Rank
        safe_options = df_filtered[df_filtered['Closing Rank'] >= rank]
        # Low chance: rank is slightly higher than Closing Rank (within a buffer of 15% or up to 15k rank)
        buffer = int(rank * 0.15) if rank < 100000 else 15000
        low_chance_options = df_filtered[(df_filtered['Closing Rank'] < rank) & (df_filtered['Closing Rank'] >= (rank - buffer))]
        
        response_text = f"Based on your profile (**{category}**, **{gender}**, **{quota}**) at **{counseling_type} {round_val}** with a rank of **{rank:,}**:\n\n"
        
        # Sort safe options ascending (meaning closest to rank first, which are higher tier)
        safe_options = safe_options.sort_values(by='Closing Rank')
        # Sort low chance options descending (meaning closest to rank first)
        low_chance_options = low_chance_options.sort_values(by='Closing Rank', ascending=False)
        
        limit = 6 if not (college or branch) else 12
        
        if college or branch:
            filter_desc = []
            if college: filter_desc.append(f"college matching **{college}**")
            if branch: filter_desc.append(f"branch matching **{branch}**")
            response_text += f"Here are the matches for " + " and ".join(filter_desc) + ":\n\n"
        
        has_low_chance = False
        
        if not safe_options.empty:
            response_text += "🟢 **Safe Options (High Probability of Admission):**\n"
            for _, row in safe_options.head(limit).iterrows():
                response_text += f"* **{row['College']}**\n"
                response_text += f"  - Branch: {row['Branch']}\n"
                response_text += f"  - Closing Rank: **{row['Closing Rank']:,}** (Status: 🟢 Safe)\n"
            response_text += "\n"
            
        if not low_chance_options.empty:
            has_low_chance = True
            response_text += "🟡 **Borderline / Low Chance Options (Slightly Above Last Year's Cutoff):**\n"
            for _, row in low_chance_options.head(limit).iterrows():
                response_text += f"* **{row['College']}**\n"
                response_text += f"  - Branch: {row['Branch']}\n"
                response_text += f"  - Closing Rank: **{row['Closing Rank']:,}** (Status: 🟡 Low Chance)\n"
            response_text += "\n"
            
        if safe_options.empty and low_chance_options.empty:
            min_cls = CHAT_DF['Closing Rank'].min()
            response_text += f"At **{rank:,} rank**, there are no options available in the {inst_type} dataset. Usually, standard NITs close around **{min_cls:,}**.\n"
            
        if has_low_chance:
            response_text += "*⚠️ **Low Chance Note:** Your rank is close to the cutoff. You might get it, but it’s not guaranteed.*\n"
            
        return jsonify({"response": response_text}), 200

    except Exception as e:
        print(f"Error in /api/chat: {e}")
        traceback.print_exc()
        return jsonify({"response": "This information is not available in the current dataset."}), 200

if __name__ == '__main__':
    # Running Flask backend
    app.run(debug=True, port=5000)


# git add .
# git commit -m "updated ui"
# git push