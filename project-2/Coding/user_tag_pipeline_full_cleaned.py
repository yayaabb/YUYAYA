import pandas as pd
import json
import requests
from tqdm import tqdm
import logging
import re
import ast

#Set logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

#Define the tag structure
tag_structure = {
    "Budget Preference": ["Low", "Medium", "High", "Luxury"],
    "Group Size": ["Solo", "Couple", "Small Group (3–5)", "Family", "Large Group(6+)"],
    "Travel Purpose": ["Relaxation", "Adventure", "Honeymoon", "Cultural", "Food", "Urban"],
    "Interest Category": ["architecture", "art", "food tours", "landmarks", "museums", "outdoors",
                          "performances", "shopping & fashions", "wildlife", "beach"],
    "Planning Style": ["well-planned", "semi-planned", "Spontaneous", "go with the flow"],
    "Transport Preference": ["Self-driving", "Train", "Bus", "Flights", "Cruise"],
    "Trip Duration Preference": ["weekend", "short(3-5d)", "long(6-14d)", "extended"]
}
EXCLUDED_FROM_SEARCH = {"Age", "Gender", "Browsing Device", "Location"}

#Value mapping for inconsistent CSV values
value_mapping = {
    "Budget Preference": {
        "Low <500£": "Low",
        "Medium 500–1500£": "Medium", 
        "High 1500–5000£": "High",
        "Luxury >5000£": "Luxury",
        "Medium 500-1500£": "Medium",
        "High 1500-5000£": "High",
        "Low <500£": "Low",
        "Luxury >5000£": "Luxury",
        "low": "Low",
        "medium": "Medium",
        "high": "High",
        "luxury": "Luxury"
        },
    "Group Size": {
        "Small Group": "Small Group (3–5)",
        "Small Group (3-5)": "Small Group (3–5)",
        "Large Group": "Large Group(6+)",
        "Large Group (6+)": "Large Group(6+)",
        "solo": "Solo",
        "couple": "Couple",
        "family": "Family"
        },
    "Trip Duration Preference": {
        "short(3–5d)": "short(3-5d)",
        "long(6–14d)": "long(6-14d)",
        "short": "short(3-5d)",
        "long": "long(6-14d)",
        "Short": "short(3-5d)",
        "Long": "long(6-14d)",
        "Weekend": "weekend",
        "Extended": "extended"
        },
    "Travel Purpose": {
        "relaxation": "Relaxation",
        "adventure": "Adventure",
        "honeymoon": "Honeymoon",
        "cultural": "Cultural",
        "food": "Food",
        "urban": "Urban"
        },
    "Planning Style": {
        "Well-planned": "well-planned",
        "Semi-planned": "semi-planned",
        "spontaneous": "Spontaneous",
        "Go with the flow": "go with the flow"
        }
}

#Build the prompt
def build_prompt(user_text):
    prompt = (
        "You are a travel classification assistant. Analyze the user's search behavior and classify them into travel preferences.\n\n"
        "CRITICAL INSTRUCTIONS:\n"
        "1. Output ONLY a valid JSON object\n"
        "2. Use EXACTLY these category names and values:\n\n"
    )
    
    for tag, options in tag_structure.items():
        options_str = ', '.join([f'"{opt}"' for opt in options])
        prompt += f'{tag}: [{options_str}]\n'
    
    prompt += (
        "\n3. IMPORTANT MAPPING GUIDELINES:\n"
        "- 'galleries' or 'art galleries' → Interest Category: \"art\"\n"
        "- 'museums' or 'science museums' → Interest Category: \"museums\"\n"
        "- 'traveling alone', 'solo', 'just me' → Group Size: \"Solo\"\n"
        "- 'couples', 'romantic', 'with my partner' → Group Size: \"Couple\"\n"
        "- 'family' → Group Size: \"Family\"\n"
        "- 'friends', 'group travel' → Group Size: \"Small Group (3–5)\" or \"Large Group(6+)\"\n"
        "- 'weekend' trips → Trip Duration Preference: \"weekend\"\n"
        "- 'short vacation', 'few days' → Trip Duration Preference: \"short(3-5d)\"\n"
        "- 'long vacation', 'extended' → Trip Duration Preference: \"long(6-14d)\" or \"extended\"\n"
        "- Budget keywords ('cheap', 'affordable') → Budget Preference: \"Low\"\n"
        "- Budget keywords ('luxury', 'premium') → Budget Preference: \"High\" or \"Luxury\"\n"
    )
    
    prompt += (
        f"\n4. User search history: \"{user_text}\"\n\n"
        "5. Output format (copy exactly, replace values):\n"
        "{\n"
    )
    for i, (tag, options) in enumerate(tag_structure.items()):
        comma = "," if i < len(tag_structure) - 1 else ""
        #default values
        if tag == "Budget Preference":
            default_option = "Medium"
        elif tag == "Group Size":
            default_option = "Solo"
        elif tag == "Planning Style":
            default_option = "well-planned"
        else:
            default_option = options[0]
        prompt += f'  "{tag}": "{default_option}"{comma}\n'
    prompt += "}\n\n"
    
    prompt += (
        "6. RULES:\n"
        "- Choose the most likely option for each category based on CLEAR search patterns\n"
        "- Use exact values from the lists above\n"
        "- Output ONLY the JSON object\n"
        "- No explanations or additional text\n"
        "- Ensure valid JSON syntax\n"
        "- Focus on travel behavior patterns, not demographics\n\n"
        "JSON OUTPUT:"
    )
    return prompt


#Clean and normalize model outputs
def clean_llama_value(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple)) and len(value) > 0:
        value = value[0]
    value = str(value).strip()
    if value.lower() in ["", "nan", "null", "none"]:
        return None
    if value.startswith('[') and value.endswith(']'):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list) and len(parsed) > 0:
                return str(parsed[0]).strip()
        except:
            return value.strip('[]').strip('\'"').strip()
    return value

#LLaMA3 inference
def run_llama3_inference(prompt, user_id=None, max_retries=2):
    for attempt in range(max_retries + 1):
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "llama3", 
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "top_p": 0.9
                    }
                },
                timeout=30
                )
            
            if response.status_code == 200:
                result = response.json()
                full_output = result.get("response", "")
                #Log raw model output
                logger.debug(f"{user_id} raw output: {full_output[:200]}")
                #Extract valid JSON
                json_result = extract_json_from_text(full_output)
                if json_result:
                    #Validate JSON structure
                    if validate_tag_structure(json_result):
                        return json_result
                    else:
                        logger.warning(f"{user_id} returned invalid JSON structure, retrying ({attempt + 1}/{max_retries + 1})")
                        if attempt < max_retries:
                            continue
                
                raise ValueError(f"Failed to extract valid JSON. Raw output: {full_output[:100]}")
            else:
                raise ValueError(f"HTTP Error: {response.status_code}")
                
        except Exception as e:
            if attempt == max_retries:
                logger.error(f"Inference failed for {user_id} after {max_retries} retries: {e}")
                return {
                    "user_id": user_id,
                    "__error__": str(e),
                    "__raw_output__": locals().get('full_output', 'No output'),
                    "__attempt__": attempt + 1
                }
            else:
                logger.warning(f"{user_id} attempt {attempt + 1} failed: {e}. Retrying")
                continue

#Extract structured JSON
def extract_json_from_text(text):
    #find a full JSON block
    json_start = text.find("{")
    json_end = text.rfind("}")
    if json_start != -1 and json_end != -1 and json_end > json_start:
        try:
            json_str = text[json_start:json_end + 1]
            result = json.loads(json_str)
            return clean_json_values(result)
        except json.JSONDecodeError:
            pass
    
    #Regex to find embedded JSON-like structure
    json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    matches = re.findall(json_pattern, text, re.DOTALL)
    
    for match in matches:
        try:
            result = json.loads(match)
            return clean_json_values(result)
        except json.JSONDecodeError:
            continue
    
    #fix common formatting issues
    if json_start != -1 and json_end != -1:
        json_str = text[json_start:json_end + 1]
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*]', ']', json_str)
        try:
            result = json.loads(json_str)
            return clean_json_values(result)
        except json.JSONDecodeError:
            pass
    
    return None

#Clean values
def clean_json_values(json_data):
    if not isinstance(json_data, dict):
        return json_data

    cleaned = {}
    for key, value in json_data.items():
        cleaned[key] = clean_llama_value(value)
    return cleaned

#Validate the JSON valid tag structure
def validate_tag_structure(json_data):
    if not isinstance(json_data, dict):
        return False
    #Compare keys against expected tags
    expected_tags = set(tag_structure.keys())
    actual_tags = set(json_data.keys()) - {"user_id"}
    #At least 30% of tags match
    overlap = len(expected_tags.intersection(actual_tags))
    return overlap >= len(expected_tags) * 0.3

#Tag value normalization
def normalize_tag_value(tag, value):
    value = clean_llama_value(value)
    if not value:
        return None
    #Apply predefined mapping
    if tag in value_mapping and value in value_mapping[tag]:
        return value_mapping[tag][value]
    if tag in tag_structure and value in tag_structure[tag]:
        return value
    value_lower = value.lower()
    for valid_option in tag_structure.get(tag, []):
        if value_lower == valid_option.lower():
            return valid_option
        if tag == "Group Size" and "small" in value_lower and "group" in value_lower:
            return "Small Group (3–5)"
        if tag == "Group Size" and "large" in value_lower and "group" in value_lower:
            return "Large Group(6+)"
    logger.warning(f"Invalid tag value: {tag} = '{value}'")
    return None

    
    #Handle list or tuple types
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return None
        value = value[0]
    
    value = str(value).strip()
    if value in ["", "nan", "NaN", "null", "None"]:
        return None
    if value.startswith('[') and value.endswith(']'):
        try:
            import ast
            parsed_list = ast.literal_eval(value)
            if isinstance(parsed_list, list) and len(parsed_list) > 0:
                value = str(parsed_list[0]).strip()
            else:
                return None
        except (ValueError, SyntaxError):
            value = value.strip('[]').strip('\'"').strip()
    if tag in value_mapping and value in value_mapping[tag]:
        return value_mapping[tag][value]
    
    #Check if the value is valid
    if tag in tag_structure and value in tag_structure[tag]:
        return value
    
    #Attempt fuzzy matching (case-insensitive and tolerant of format variation)
    if tag in tag_structure:
        value_lower = value.lower()
        for valid_option in tag_structure[tag]:
            if value_lower == valid_option.lower():
                return valid_option
            if tag == "Group Size" and "small" in value_lower and "group" in value_lower:
                return "Small Group (3–5)"
            if tag == "Group Size" and "large" in value_lower and "group" in value_lower:
                return "Large Group(6+)"
    
    #Log invalid
    logger.warning(f"Invalid tag value {tag} = '{value}'")
    return None2

#Load user profile
def load_user_basic_info(basic_info_path):
    try:
        df_basic = pd.read_csv(basic_info_path)
        user_basic_dict = {}
        
        for _, row in df_basic.iterrows():
            user_id = row.get('user')
            if pd.isna(user_id):
                continue
                
            user_info = {}
            for tag in tag_structure.keys():
                if tag in row:
                    raw_value = row[tag]
                    normalized_value = normalize_tag_value(tag, raw_value)
                    if normalized_value:
                        user_info[tag] = normalized_value
            
            if user_info:
                user_basic_dict[user_id] = user_info
        
        logger.info(f"Successfully loaded basic information for {len(user_basic_dict)} users.")
        return user_basic_dict
    
    except Exception as e:
        logger.warning(f"Failed to load basic user info: {e}. Proceeding with search records only.")
        return {}

#Evidence validation
def validate_search_evidence(tag, basic_value, search_value, search_text):
    #If no search text is available, default to trusting the inferred value
    if not search_text:
        return True
    
    search_lower = search_text.lower()
    if tag == "Group Size":
        group_evidence = {
            "Solo": ['alone', 'solo', 'just me', 'by myself', 'traveling alone', 'individual'],
            "Couple": ['couple', 'romantic', 'with my partner', 'two people', 'honeymoon'],
            "Small Group (3–5)": ['friends', 'small group', 'few friends', 'group travel', 'friends trip'],
            "Family": ['family', 'kids', 'children', 'parents', 'family trip'],
            "Large Group(6+)": ['large group', 'big group', 'big family', 'many people', 'group of']
        }
        
        #Check the strength of evidence
        basic_evidence_count = sum(1 for keyword in group_evidence.get(basic_value, []) 
                                 if keyword in search_lower)
        search_evidence_count = sum(1 for keyword in group_evidence.get(search_value, []) 
                                  if keyword in search_lower)
        
        #If basic value has stronger evidence, keep the original value
        if basic_evidence_count > search_evidence_count:
            return False
            
    elif tag == "Travel Purpose":
        purpose_evidence = {
            "Relaxation": ['relaxation', 'relax', 'peaceful', 'calm', 'rest'],
            "Adventure": ['adventure', 'exciting', 'thrill', 'extreme'],
            "Honeymoon": ['honeymoon', 'romantic', 'couple'],
            "Cultural": ['cultural', 'culture', 'heritage', 'history', 'traditional'],
            "Food": ['food', 'foodie', 'dining', 'restaurant', 'cuisine', 'local dishes'],
            "Urban": ['urban', 'city', 'metropolitan', 'downtown']
        }
        
        basic_evidence_count = sum(1 for keyword in purpose_evidence.get(basic_value, []) 
                                 if keyword in search_lower)
        search_evidence_count = sum(1 for keyword in purpose_evidence.get(search_value, []) 
                                  if keyword in search_lower)
        
        if basic_evidence_count > search_evidence_count:
            return False
            
    elif tag == "Planning Style":
        planning_evidence = {
            "well-planned": ['well-planned', 'planned', 'organized', 'structured', 'detailed'],
            "semi-planned": ['semi-planned', 'partially planned', 'some planning'],
            "Spontaneous": ['spontaneous', 'last-minute', 'impromptu', 'unplanned'],
            "go with the flow": ['go with the flow', 'flexible', 'adaptable', 'casual']
        }
        
        basic_evidence_count = sum(1 for keyword in planning_evidence.get(basic_value, []) 
                                 if keyword in search_lower)
        search_evidence_count = sum(1 for keyword in planning_evidence.get(search_value, []) 
                                  if keyword in search_lower)
        
        if basic_evidence_count > search_evidence_count:
            return False
    
    return True 

# Detect and resolve conflicts between basic user info and inferred tags from search behavior
# Rule: a true conflict only occurs when the basic info already contains a value, the inferred value is different,
# and there is strong supporting evidence from the search record
# Exclude certain tags that should not be inferred from search data
def resolve_conflicts(user_id, basic_info, search_tags, user_search_text=""):
    conflicts = []
    supplements = []
    rejected_conflicts = []
    updated_info = basic_info.copy()
    
    for tag, search_value in search_tags.items():
        if tag == "user_id" or tag.startswith("__"):
            continue
        #Skip excluded tag
        if tag in EXCLUDED_FROM_SEARCH:
            logger.debug(f"Skipping excluded tag: {tag}")
            continue
        
        #Normalize the inferred search value
        normalized_search_value = normalize_tag_value(tag, search_value)
        if not normalized_search_value:
            logger.warning(f"{user_id} has an invalid tag value from search results: {tag} = '{search_value}'")
            continue
        #If tag exists in basic info and is not empty
        if tag in basic_info and basic_info[tag]: 
            basic_value = basic_info[tag]
            if basic_value != normalized_search_value:
                # Check if search text provides strong enough evidence for the new value
                is_valid_inference = validate_search_evidence(tag, basic_value, normalized_search_value, user_search_text)
                if is_valid_inference:
                    conflicts.append({
                        'tag': tag,
                        'basic_value': basic_value,
                        'search_value': normalized_search_value,
                        'action': 'conflict_updated'
                    })
                    updated_info[tag] = normalized_search_value
                else:
                    rejected_conflicts.append({
                        'tag': tag,
                        'basic_value': basic_value,
                        'search_value': normalized_search_value,
                        'action': 'rejected_weak_evidence'
                    })
                    logger.info(f"{user_id} rejected weak evidence for conflict: {tag} = '{basic_value}' (inferred: '{normalized_search_value}')")

        else:
            supplements.append({
                'tag': tag,
                'basic_value': basic_info.get(tag, None),
                'search_value': normalized_search_value,
                'action': 'supplemented'
            })
            updated_info[tag] = normalized_search_value
    all_changes = conflicts + supplements + rejected_conflicts
    
    return updated_info, all_changes, len(conflicts), len(supplements), len(rejected_conflicts)

#Generate conflict report
def generate_detailed_conflict_report(all_conflicts):
    if not all_conflicts:
        return "No conflicts found"
    
    report = []
    report.append(f"A total of {len(all_conflicts)} users with conflicts were processed.")
    
    #Count different types of conflict actions
    conflict_stats = {}
    update_details = {}
    add_details = {}
    
    for user_id, user_conflicts in all_conflicts.items():
        for conflict in user_conflicts:
            tag = conflict['tag']
            action = conflict['action']
            key = f"{tag}_{action}"
            conflict_stats[key] = conflict_stats.get(key, 0) + 1
            
            if action == 'updated':
                if tag not in update_details:
                    update_details[tag] = {}
                change_key = f"{conflict['basic_value']} -> {conflict['search_value']}"
                update_details[tag][change_key] = update_details[tag].get(change_key, 0) + 1
            elif action == 'added':
                if tag not in add_details:
                    add_details[tag] = {}
                add_details[tag][conflict['search_value']] = add_details[tag].get(conflict['search_value'], 0) + 1
    
    report.append("\nConflict Overview")
    for key, count in sorted(conflict_stats.items()):
        tag, action = key.rsplit('_', 1)
        report.append(f"  {tag} ({action}): {count}")
    
    if update_details:
        report.append("\nUpdate Details")
        for tag, changes in update_details.items():
            report.append(f"\n{tag}:")
            for change, count in sorted(changes.items()):
                report.append(f"  {change}: {count}")
    
    if add_details:
        report.append("\nAddition Details")
        for tag, values in add_details.items():
            report.append(f"\n{tag}:")
            for value, count in sorted(values.items()):
                report.append(f"  {value}: {count}")
    
    return "\n".join(report)

#Main process
def process_users(search_csv_path, output_csv_path, basic_info_path=None, 
                 conflicts_output_path=None):
    #Load search records and user info
    df_search = pd.read_csv(search_csv_path)
    user_search_dict = df_search.groupby("user_id")["search_text"].apply(lambda x: " ".join(x)).to_dict()
    user_basic_info = {}
    if basic_info_path:
        user_basic_info = load_user_basic_info(basic_info_path)
    all_results = []
    all_conflicts = {}
    
    for user_id, text in tqdm(user_search_dict.items(), desc="🔍 Processing users"):
        #Extract tags from search records
        prompt = build_prompt(text)
        search_tags = run_llama3_inference(prompt, user_id)
        if not search_tags or "__error__" in search_tags:
            all_results.append(search_tags)
            continue
        basic_info = user_basic_info.get(user_id, {})
        #Resolve conflicts and update info
        if basic_info:
            updated_info, all_changes, conflict_count, supplement_count, rejected_count = resolve_conflicts(user_id, basic_info, search_tags, text)
            if all_changes:
                all_conflicts[user_id] = all_changes
                if conflict_count > 0:
                    logger.info(f"{user_id} had {conflict_count} real conflicts, {supplement_count} supplements, and {rejected_count} weak conflicts rejected")
                    conflicts_only = [c for c in all_changes if c['action'] == 'conflict_updated']
                    for conflict in conflicts_only:
                        logger.info(f"Conflict detail: {conflict['tag']} = '{conflict['basic_value']}' -> '{conflict['search_value']}'")
                else:
                    if rejected_count > 0:
                        logger.info(f" {user_id} added {supplement_count} supplements and rejected {rejected_count} weak conflicts")
                    else:
                        logger.debug(f"User {user_id} added {supplement_count} supplements")
        else:
            updated_info = search_tags
        
        updated_info["user_id"] = user_id
        all_results.append(updated_info)
    
    #Save result and conflict log
    df_result = pd.DataFrame(all_results)
    df_result.to_csv(output_csv_path, index=False)
    logger.info(f"Tag extraction completed. Output saved to: {output_csv_path}")
    
    if conflicts_output_path and all_conflicts:
        conflict_records = []
        for user_id, conflicts in all_conflicts.items():
            for conflict in conflicts:
                conflict_records.append({
                    'user_id': user_id,
                    'tag': conflict['tag'],
                    'basic_value': conflict['basic_value'],
                    'search_value': conflict['search_value'],
                    'action': conflict['action']
                })
        
        df_conflicts = pd.DataFrame(conflict_records)
        df_conflicts.to_csv(conflicts_output_path, index=False)
        logger.info(f"Conflict log saved to：{conflicts_output_path}")
    
    #ensure all basic user info is included
    logger.info("Ensuring all basic user information is preserved")
    basic_users = user_basic_info
    updated_users = {entry['user_id']: entry for entry in all_results if 'user_id' in entry}
    for user_id, basic_info in basic_users.items():
        if user_id not in updated_users:
            # No search record, retain basic info
            updated_users[user_id] = basic_info.copy()
            logger.debug(f" {user_id} has no search records, retaining basic info")
        else:
            #keep any missing basic fields
            for key, value in basic_info.items():
                if key not in updated_users[user_id] or updated_users[user_id][key] == "":
                    updated_users[user_id][key] = value
    report = generate_detailed_conflict_report(all_conflicts, sample_size=5)
    logger.info(f"\nDetailed conflict resolution report:\n{report}")
    
    return df_result, all_conflicts


if __name__ == "__main__":
    process_users(
        search_csv_path=" User Profiles/Optimized_Search_Queries.csv",
        output_csv_path=" User Profiles/user_tag_outputs_merged.csv",
        basic_info_path=" User Profiles/Simulated_Unprocessed_User_Info.csv",
        conflicts_output_path="user_conflicts_log.csv"
        )
    
