import requests
import json
import re

def extract_json_str(s: str) -> str:
    """
    Finds the first {...} substring in s and returns it, else returns s unchanged.
    """
    # Greedily capture from the first “{” to the last “}”
    match = re.search(r"\{.*\}", s, flags=re.DOTALL)
    return match.group(0) if match else s

def call_hf_api(prompt, api_key, model='mistralai/Mixtral-8x7B-Instruct-v0.1'):
    url = f"https://api-inference.huggingface.co/models/{model}"
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {"inputs": prompt}
    response = requests.post(url, headers=headers, json=payload)
    data = response.json()
    # Hugging Face inference API can return either a list of outputs or a dict
    if isinstance(data, list) and len(data) > 0 and 'generated_text' in data[0]:
        return data[0]['generated_text']
    elif isinstance(data, dict) and 'generated_text' in data:
        return data['generated_text']
    # Fallback: if the API returns a nested dict under 'choices'
    if isinstance(data, dict) and 'choices' in data:
        choice = data['choices'][0]
        return choice.get('text') or choice.get('generated_text') or ""
    # If nothing matched, convert entire payload to string
    return str(data)



def prompt_analysis(convs, api_key, task):
    agent_ids, borrower_ids, violating_ids = [], [], []
    for cid, utts in convs.items():
        transcript = "\n".join(f"{u['speaker']}: {u['text']}" for u in utts)

        if task == "Profanity":
            # Agent prompt
            prompt_a = (
                "You are a compliance reviewer.  \n"
                "Read the following debt-collection call transcript and answer as JSON with two fields:\n"
                "  - answer: \"Yes\" or \"No\"  \n"
                "  - explanation: one-sentence rationale quoting the offending phrase if any\n\n"
                f"Transcript:\n{transcript}\n\n"
                "Question: Does the **agent** use profanity in this call?  \n"
                "Only return a valid JSON object, nothing else. No need to mention the question or prompt again."
            )
            raw = call_hf_api(prompt_a, api_key)
            trimmed = extract_json_str(raw)
            print(trimmed)
            try:
                obj = json.loads(trimmed)
            except json.JSONDecodeError:
                obj = {"answer":"No","explanation":"Parsing error—assuming no profanity."}
            if obj["answer"].lower()=="yes":
                agent_ids.append((cid, obj["explanation"]))

            # Borrower prompt
            prompt_b = (
                "You are a compliance reviewer.  \n"
                "Read the following debt-collection call transcript and answer as JSON with two fields:\n"
                "  • answer: \"Yes\" or \"No\"  \n"
                "  • explanation: one-sentence rationale quoting the offending phrase if any\n\n"
                f"Transcript:\n{transcript}\n\n"
                "Question: Does the **borrower** use profanity in this call?  \n"
                "Only return a valid JSON object, nothing else. No need to mention the question or prompt again."
            )
            raw_b = call_hf_api(prompt_b, api_key)
            trimmed_b = extract_json_str(raw_b)
            print(trimmed_b)
            try:
                obj = json.loads(trimmed_b)
            except json.JSONDecodeError:
                obj = {"answer":"No","explanation":"Parsing error—assuming no profanity."}
            if obj["answer"].lower()=="yes":
                borrower_ids.append((cid, obj["explanation"]))

        else:
            # Privacy prompt
            prompt = (
                        "You are a compliance reviewer.  \n"
                        "I will give you a debt-collection call transcript.  \n"
                        "First, find the earliest agent utterance that **verifies identity**, "
                        "e.g. asks for date of birth, address, or SSN.  \n"
                        "Next, find the earliest agent utterance that **shares sensitive details**, "
                        "e.g. mentions balance, account number, debt amount, statement.  \n"
                        "If the sensitive-details utterance comes **before** the identity-verification utterance, "
                        "this is a violation. Otherwise it is compliant.  \n\n"
                        f"Transcript:\n{transcript}\n\n"
                        "Read the following debt-collection call transcript and answer as JSON with two fields:\n"
                        "  - answer: \"Yes\" or \"No\"  \n"
                        "  - explanation: one-sentence rationale quoting the compliancy violation phrase if any\n\n"
                        "Return only the JSON object and nothing else."
                        )
            raw = call_hf_api(prompt, api_key)
            trimmed = extract_json_str(raw)
            print(trimmed)
            try:
                obj = json.loads(trimmed)
            except json.JSONDecodeError:
                obj = {"answer":"No","explanation":"Parsing error—assuming compliant."}
            if obj["answer"].lower()=="yes":
                violating_ids.append((cid, obj["explanation"]))

    # Return list of tuples (call_id, explanation) so UI can use call_id as dropdown key
    if task == "Profanity":
        return agent_ids, borrower_ids
    else:
        return violating_ids
