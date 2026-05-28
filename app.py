import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import traceback
import random

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


app = Flask(__name__)
CORS(app)  # Enable CORS for frontend requests

USER_CONTEXT = {
    "last_rank": None,
    "last_category": None,
    "last_gender": None,
    "last_quota": None,
    "last_college": None,
    "last_branch": None,
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
                    temp_df = pd.read_csv(fname)
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
                df = pd.read_csv(filename)
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
                temp_df = pd.read_csv(fname)
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
                    temp_df = pd.read_csv(fname)
                    temp_df['Institute_Type'] = itype
                    dfs.append(temp_df)
                except Exception as e:
                    print(f"Error loading {fname}: {e}")
        else:
            fname = file_map.get(institute_type)
            if not fname:
                return jsonify({"error": "Invalid institute type."}), 400
            try:
                temp_df = pd.read_csv(fname)
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
                temp_df = pd.read_csv(fname)
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
        "trichy": "Trichy",
        "tiruchirappalli": "Trichy",
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
        "branch": extracted_branch
    }

def load_all_cutoff_data():
    dfs = []
    
    # 1. Look for main nit_data.csv in current directory
    if os.path.exists('nit_data.csv'):
        try:
            df = pd.read_csv('nit_data.csv')
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
                                df = pd.read_csv(path)
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

def get_casual_response(message):
    msg_lower = message.lower()
    global USER_CONTEXT
    
    rank = USER_CONTEXT.get("last_rank")
    category = USER_CONTEXT.get("last_category", "OPEN")
    college = USER_CONTEXT.get("last_college")
    branch = USER_CONTEXT.get("last_branch")
    
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
            
        if CHAT_DF is None:
            CHAT_DF = load_all_cutoff_data()
            
        if CHAT_DF is None:
            return jsonify({"response": "This information is not available in the current dataset."}), 200

        # Parse message
        parsed = parse_chat_message(message)
        rank = parsed.get('rank')
        category = parsed.get('category')
        gender = parsed.get('gender')
        quota = parsed.get('quota')
        college = parsed.get('college')
        branch = parsed.get('branch')
        
        # Keep track of parsed parameters in session context
        global USER_CONTEXT
        try:
            if rank: USER_CONTEXT['last_rank'] = rank
            if category: USER_CONTEXT['last_category'] = category
            if gender: USER_CONTEXT['last_gender'] = gender
            if quota: USER_CONTEXT['last_quota'] = quota
            if college: USER_CONTEXT['last_college'] = college
            if branch: USER_CONTEXT['last_branch'] = branch
        except Exception:
            pass
            
        current_rank = USER_CONTEXT.get('last_rank')
        current_category = USER_CONTEXT.get('last_category')
        current_quota = USER_CONTEXT.get('last_quota')
        current_gender = USER_CONTEXT.get('last_gender')
        
        # Check if this matches a casual intent first
        casual_response = get_casual_response(message)
        is_direct_cutoff = (rank is not None) or (college is not None) or (branch is not None)
        
        if casual_response and not is_direct_cutoff:
            return jsonify({"response": casual_response}), 200
            
        # Determine if this is a cutoff prediction query or context update
        is_cutoff_query = is_direct_cutoff or (current_rank is not None)
        
        if is_cutoff_query:
            # Enforce completeness check for Rank, Category, Quota, Gender
            if current_rank is not None:
                missing = []
                if not current_category: missing.append("Category (OPEN, OBC-NCL, SC, ST, EWS, or PwD categories)")
                if not current_quota: missing.append("Quota (Home State or Other State)")
                if not current_gender: missing.append("Gender (Gender-Neutral or Female-only)")
                
                if missing:
                    missing_str = " and ".join([", ".join(missing[:-1]), missing[-1]] if len(missing) > 1 else missing)
                    responses = [
                        f"I see your rank is **{current_rank:,}**. To give you 100% accurate college options, could you please tell me your **{missing_str}**? Cutoffs vary significantly based on these details!",
                        f"At rank **{current_rank:,}**, predictions depend heavily on your profile details. Could you share your **{missing_str}** so I can search the official JoSAA/CSAB cutoffs accurately?",
                        f"Got the rank **{current_rank:,}**! 🎓 Before I show you the matching colleges, I'll need your **{missing_str}** to make sure we get the correct data. What are those?"
                    ]
                    return jsonify({"response": random.choice(responses)}), 200
        else:
            # Custom guiding prompt that also respects memory
            prev_rank = USER_CONTEXT.get('last_rank')
            prev_category = USER_CONTEXT.get('last_category')
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
                
        # Re-assign validated consolidated values for prediction engine
        category = current_category
        gender = current_gender
        quota = current_quota
        rank = current_rank
        
        # Filter matching rows
        df_filtered = CHAT_DF.copy()
        
        # Filter Category (Required matching)
        if category:
            cat_variants = get_category_variants(category)
            df_filtered = df_filtered[df_filtered['Category'].isin(cat_variants)]
            
        # Filter Gender
        if gender == 'Female-only':
            df_filtered = df_filtered[df_filtered['Gender'].isin(['Female-only', 'Gender-Neutral'])]
        else:
            df_filtered = df_filtered[df_filtered['Gender'] == 'Gender-Neutral']
            
        # Filter Quota (If user specified OS, HS, or AI)
        if quota:
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
        if rank:
            df_filtered = df_filtered.sort_values(by='Closing Rank')
            
            # Safe: rank <= Closing Rank
            safe_options = df_filtered[df_filtered['Closing Rank'] >= rank]
            # Low chance: rank is slightly higher than Closing Rank (within a buffer of 15% or up to 15k rank)
            buffer = int(rank * 0.15) if rank < 100000 else 15000
            low_chance_options = df_filtered[(df_filtered['Closing Rank'] < rank) & (df_filtered['Closing Rank'] >= (rank - buffer))]
            
            response_text = f"Based on your profile (**{category}**, **{gender}**"
            if quota:
                response_text += f", **{quota}**"
            response_text += f") and your rank of **{rank:,}**:\n\n"
            
            if college:
                matched_college = df_filtered['College'].iloc[0]
                response_text += f"Looking at **{matched_college}**:\n"
                
                if branch:
                    matched_branch = df_filtered['Branch'].iloc[0]
                    records = df_filtered[df_filtered['Branch'].str.contains(branch, case=False, na=False)]
                    if not records.empty:
                        closing_rank = int(records['Closing Rank'].max())
                        op_rank = int(records['Opening Rank'].min()) if not pd.isna(records['Opening Rank'].min()) else None
                        
                        if rank <= closing_rank:
                            response_text += f"🎉 **Yes, you have an excellent chance!** last year's cutoff for **{matched_branch}** was between Opening Rank **{op_rank:,}** and Closing Rank **{closing_rank:,}**. You are well within this range."
                        elif rank <= (closing_rank + buffer):
                            response_text += f"⚖️ **Borderline/Low Chance.** Cutoffs for **{matched_branch}** closed around **{closing_rank:,}** last year. At your rank of **{rank:,}**, you are close, so it stands as a low chance option in later rounds if cutoffs shift."
                        else:
                            response_text += f"❌ **Unlikely.** Last year's cutoff for **{matched_branch}** closed at **{closing_rank:,}**. Since your rank is **{rank:,}**, this option is out of range."
                    else:
                        response_text += f"No records found for branch **{branch}** at this college."
                else:
                    college_safe = safe_options[safe_options['College'].str.contains(college, case=False, na=False)]
                    college_low = low_chance_options[low_chance_options['College'].str.contains(college, case=False, na=False)]
                    
                    if not college_safe.empty:
                        branches = college_safe['Branch'].unique()[:3]
                        br_list = ", ".join(branches)
                        max_closing = college_safe['Closing Rank'].max()
                        response_text += f"🎉 **You have safe chances here!** You can comfortably unlock several branches like **{br_list}**. The lowest cutoff closed around **{max_closing:,}**."
                    elif not college_low.empty:
                        br = college_low['Branch'].iloc[0]
                        cls = college_low['Closing Rank'].max()
                        response_text += f"⚖️ **Low Chance / Borderline.** You are slightly above last year's cutoffs for this college. The closest option is **{br}** which closed around **{cls:,}**."
                    else:
                        min_closing = df_filtered['Closing Rank'].min()
                        response_text += f"❌ **Unlikely.** Cutoffs for this college closed around **{min_closing:,}** (highest closing rank was **{df_filtered['Closing Rank'].max():,}**). At your rank of **{rank:,}**, it looks difficult."
            else:
                if not safe_options.empty:
                    response_text += f"🎉 **Safe Options (Comfortable Seats):**\n"
                    safe_samples = safe_options.tail(3)
                    for _, row in safe_samples.iterrows():
                        response_text += f"* **{row['College']}** - {row['Branch']} (Closed at **{row['Closing Rank']:,}**)\n"
                
                if not low_chance_options.empty:
                    response_text += f"\n⚖️ **Borderline / Low Chance Options:**\n"
                    low_samples = low_chance_options.head(3)
                    for _, row in low_samples.iterrows():
                        response_text += f"* **{row['College']}** - {row['Branch']} (Closed at **{row['Closing Rank']:,}**)\n"
                        
                if safe_options.empty and low_chance_options.empty:
                    min_cls = CHAT_DF['Closing Rank'].min()
                    response_text += f"At **{rank:,} rank**, there are no options available in the NIT/IIIT dataset. Usually, standard NITs close around **{min_cls:,}**."
            
            return jsonify({"response": response_text}), 200

        else:
            response_text = f"Based on your filters (**{category}**, **{gender}**):\n\n"
            
            if college:
                matched_college = df_filtered['College'].iloc[0]
                response_text += f"For **{matched_college}**:\n"
                
                if branch:
                    records = df_filtered[df_filtered['Branch'].str.contains(branch, case=False, na=False)]
                    if not records.empty:
                        row = records.iloc[0]
                        response_text += f"The cutoff for **{row['Branch']}** opened at **{row['Opening Rank']:,}** and closed at **{row['Closing Rank']:,}** last year."
                    else:
                        response_text += f"I couldn't find cutoffs for the branch **{branch}** at this college."
                else:
                    df_sorted = df_filtered.sort_values(by='Closing Rank')
                    response_text += "Here are the closing ranks for popular branches last year:\n"
                    for _, row in df_sorted.head(4).iterrows():
                        response_text += f"* **{row['Branch']}**: Closed at **{row['Closing Rank']:,}**\n"
            else:
                if branch:
                    response_text += f"Here are some cutoffs for **{branch}** across top institutes last year:\n"
                    df_sorted = df_filtered.sort_values(by='Closing Rank')
                    for _, row in df_sorted.head(4).iterrows():
                        response_text += f"* **{row['College']}** - Closed at **{row['Closing Rank']:,}**\n"
                else:
                    response_text = "Hello! I am your smart JEE Counseling Assistant. 🎓\n\nHow can I help you today? Please tell me your **JEE Main Rank**, **Category**, and **Gender** or ask about a specific college/branch (e.g., *'Can I get NIT Jaipur at 70k rank?'* or *'Safe options at 40000 rank OPEN SC'*)."

            return jsonify({"response": response_text}), 200

    except Exception as e:
        print(f"Error in /api/chat: {e}")
        traceback.print_exc()
        return jsonify({"response": "This information is not available in the current dataset."}), 200

if __name__ == '__main__':
    # Running Flask backend
    app.run(debug=True, port=5000)
