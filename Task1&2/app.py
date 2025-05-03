import streamlit as st
from utils import load_json_convs, annotate_with_similarity_orderaware, PROF_PATTERN, VERIFY, SENSITIVE
from ml_models import train_ml_with_similarity
from llm_prompt import prompt_analysis

st.set_page_config(page_title="Debt Collection Analyzer", layout="wide")
st.title("📞 Debt Collection Compliance Analyzer")

uploaded_files = st.file_uploader("Upload conversation JSON files", type=["json"], accept_multiple_files=True)
approach = st.selectbox("Select Analysis Approach", ["Pattern Matching", "Machine Learning", "Prompt LLM"])
task = st.selectbox("Select Task", ["Profanity", "Privacy"])
api_key = st.text_input("HuggingFace API Key", type="password") if approach == "Prompt LLM" else None

if uploaded_files:
    convs = load_json_convs(uploaded_files)
    if approach == "Pattern Matching":
        agent_ids = []
        borrower_ids = []
        violating_ids = []
        for cid, utts in convs.items():
            verified = False
            for utt in utts:
                txt = utt['text']; speaker = utt['speaker'].lower()
                if task == "Profanity" and PROF_PATTERN.search(txt):
                    if speaker.startswith("agent"): agent_ids.append(cid); break
                    else: borrower_ids.append(cid); break
                elif task == "Privacy" and speaker.startswith("agent"):
                    if any(v in txt.lower() for v in VERIFY): verified = True
                    if any(s in txt.lower() for s in SENSITIVE) and not verified:
                        violating_ids.append(cid); break
        if task == "Profanity":
            st.write("Agent Profanity Call IDs:", agent_ids)
            st.write("Borrower Profanity Call IDs:", borrower_ids)
        else:
            st.write("Privacy Violating Call IDs:", violating_ids)

    elif approach == "Machine Learning":
        model = train_ml_with_similarity(convs, task.lower())
        if model is None:
            st.warning("Insufficient label variety to train ML model—falling back to Pattern Matching.")
            # Optionally rerun pattern matching here:
            # agent_ids, borrower_ids, violating_ids = pattern_match_detect(convs)
        else:
            detected = []
            for cid, utts in convs.items():
                if any(model.predict([utt["text"]])[0] == 1 for utt in utts):
                    detected.append(cid)

            if task == "Profanity":
                st.write("Call IDs likely showing profanity:", detected)
            else:
                st.write("Privacy Violating Call IDs:", detected)

    elif approach == "Prompt LLM":
        if not api_key:
            st.warning("Please provide a HuggingFace API Key.")
        else:
            # prompt_analysis now returns lists of (call_id, explanation)
            results = prompt_analysis(convs, api_key, task)

            if task == "Profanity":
                agent_list, borrower_list = results

                st.subheader("Agent Profanity")
                if agent_list:
                    # dropdown of agent-flagged calls
                    sel_agent = st.selectbox(
                        "Select a call with agent profanity",
                        agent_list,
                        format_func=lambda x: x[0]
                    )
                    st.markdown("**Explanation:**")
                    st.write(sel_agent[1])
                else:
                    st.write("No agent profanity detected.")

                st.subheader("Borrower Profanity")
                if borrower_list:
                    sel_borrower = st.selectbox(
                        "Select a call with borrower profanity",
                        borrower_list,
                        format_func=lambda x: x[0]
                    )
                    st.markdown("**Explanation:**")
                    st.write(sel_borrower[1])
                else:
                    st.write("No borrower profanity detected.")

            else:  # Privacy
                viol_list = results  # list of (call_id, explanation)

                st.subheader("Privacy Violations")
                if viol_list:
                    sel_priv = st.selectbox(
                        "Select a call with privacy violation",
                        viol_list,
                        format_func=lambda x: x[0]
                    )
                    st.markdown("**Explanation:**")
                    st.write(sel_priv[1])
                else:
                    st.write("No privacy violations detected.")

