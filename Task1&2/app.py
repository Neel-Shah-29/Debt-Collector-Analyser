import streamlit as st
from utils import load_json_convs, PROF_PATTERN, VERIFY, SENSITIVE
from ml_models import train_ml_with_similarity
from llm_prompt import PROMPT_RESULTS, prompt_analysis, call_hf_api
import pandas as pd
import re,json

FINAL_RESULTS: list[dict] = []

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
            # Run prompt_analysis once and cache responses
            prompt_analysis(convs, api_key, task)

            # Now display results from PROMPT_RESULTS
            if task == "Profanity":
                agent = [(cid, expl) for cid, t, role, expl in PROMPT_RESULTS
                         if t=="Profanity" and role=="agent" and expl]
                borrower = [(cid, expl) for cid, t, role, expl in PROMPT_RESULTS
                            if t=="Profanity" and role=="borrower" and expl]

                st.subheader("Agent Profanity")
                if agent:
                    sel = st.selectbox("Select agent call", agent, format_func=lambda x: x[0])
                    st.markdown("**Explanation:**"); st.write(sel[1])
                else:
                    st.write("No agent profanity detected.")

                st.subheader("Borrower Profanity")
                if borrower:
                    sel = st.selectbox("Select borrower call", borrower, format_func=lambda x: x[0])
                    st.markdown("**Explanation:**"); st.write(sel[1])
                else:
                    st.write("No borrower profanity detected.")

            else:  # Privacy
                viol = [(cid, expl) for cid, t, role, expl in PROMPT_RESULTS
                        if t=="Privacy" and role=="agent" and expl]

                st.subheader("Privacy Violations")
                if viol:
                    sel = st.selectbox("Select violating call", viol, format_func=lambda x: x[0])
                    st.markdown("**Explanation:**"); st.write(sel[1])
                else:
                    st.write("No privacy violations detected.")
                    
            df_cache = pd.DataFrame(
                [ {"Call ID": cid, "Task": t, "Role": role, "Explanation": expl}
                for cid, t, role, expl in PROMPT_RESULTS
                if t == task ]
            )
            if not df_cache.empty:
                st.download_button(
                    "📥 Download raw Prompt‐LLM results as CSV",
                    df_cache.to_csv(index=False),
                    file_name="prompt_llm_cache.csv",
                    mime="text/csv")

    
# -- Agentic AI Prompt section at the bottom --
st.markdown("---")
st.subheader("🛠️ Agentic AI Orchestrating Prompt")
run = st.button("Run Agentic AI Final Decision")

if run:
    if not api_key:
        st.info("Provide a HuggingFace API Key above to run the Agentic AI Prompt.")
    else:
        rows = []
        # prepare ML models
        ml_p = train_ml_with_similarity(convs, "profanity")
        ml_v = train_ml_with_similarity(convs, "privacy")

        # build lookup from PROMPT_RESULTS
        llm_map_p = {cid: expl for cid, t, role, expl in PROMPT_RESULTS if t=="Profanity" and role=="agent"}
        llm_map_v = {cid: expl for cid, t, role, expl in PROMPT_RESULTS if t=="Privacy" and role=="agent"}

        for cid, utts in convs.items():
            transcript = "\n".join(f"{u['speaker']}: {u['text']}" for u in utts)
            # signals for Profanity
            pat_p = any(PROF_PATTERN.search(u["text"]) for u in utts)
            ml_flag_p = any(ml_p.predict([u["text"]])[0] == 1 for u in utts) if ml_p else False
            llm_flag_p = cid in llm_map_p
            # signals for Privacy
            verified = False
            pat_v = False
            for u in utts:
                t, sp = u["text"].lower(), u["speaker"].lower()
                if sp.startswith("agent") and any(v in t for v in VERIFY):
                    verified = True
                if sp.startswith("agent") and any(s in t for s in SENSITIVE) and not verified:
                    pat_v = True
                    break
            ml_flag_v = any(ml_v.predict([u["text"]])[0] == 1 for u in utts) if ml_v else False
            llm_flag_v = cid in llm_map_v

            # Final orchestrating prompt: only decision + reason
            orchestrate = f"""
                You are the final arbiter for call {cid}.
                Signals:
                PROF → Pattern={pat_p}, ML={ml_flag_p}, LLM={llm_flag_p}
                PRIV → Pattern={pat_v}, ML={ml_flag_v}, LLM={llm_flag_v}

                Transcript:
                {transcript}

                "Based on these signals, and weighting Pattern=25%, ML=25%, LLM=50%,\n"
                "return a JSON object with:\n"
                "  answer: \"Yes\" or \"No\"\n"
                "  explanation: one-sentence rationale for your final decision of person willing to pay the debt or not.\n"
                "Only return the JSON object."
                """
            out = call_hf_api(orchestrate, api_key)
        
            m = re.search(r"\{.*\}", out, flags=re.DOTALL)
            try:
                final = json.loads(m.group(0) if m else out)
            except:
                final = {
                    "answer": "No",
                    "explanation": "Parsing error."
                }

            # Append to global cache
            FINAL_RESULTS.append({
                "Call ID": cid,
                "Predicted Outcome": final.get("answer", "No"),
                "Explanation": final.get("explanation", ""),
            })

        # After the loop, build DataFrame from FINAL_RESULTS
        df_final = pd.DataFrame(FINAL_RESULTS)

        # Download button
        st.download_button(
            "📥 Download Agentic AI Decisions",
            df_final.to_csv(index=False),
            "agentic_ai_decisions.csv",
            "text/csv"
        )
        st.dataframe(df_final, use_container_width=True)
            
            
    