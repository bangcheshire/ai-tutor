import streamlit as st
import base64
import time
import uuid
import pandas as pd
import json
import re
from openai import OpenAI


# ==========================================
# 1. Basic Configuration
# ==========================================
st.set_page_config(page_title="AI Learning Companion", page_icon="🎓", layout="wide")

st.markdown("""
<style>
section[data-testid="stSidebar"] .stButton > button {
    height: 2.3rem !important;
    min-height: 2.3rem !important;
    padding: 0 10px !important;
    font-size: 0.87rem !important;
    border-radius: 8px !important;
    transition: all 0.15s ease !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
}
section[data-testid="stSidebar"] .stButton > button[kind="primary"] {
    background-color: #1a73e8 !important;
    border-color:     #1a73e8 !important;
    color:            #ffffff !important;
    text-align:       left    !important;
    justify-content:  flex-start !important;
}
section[data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {
    background-color: #1557b0 !important;
    border-color:     #1557b0 !important;
}
section[data-testid="stSidebar"] .stButton > button[kind="secondary"] {
    background-color: transparent !important;
    border-color:     #e0e0e0   !important;
    color:            #444444   !important;
    text-align:       left      !important;
    justify-content:  flex-start !important;
}
section[data-testid="stSidebar"] .stButton > button[kind="secondary"]:hover {
    background-color: #e8f0fe !important;
    border-color:     #1a73e8 !important;
    color:            #1a73e8 !important;
}
section[data-testid="stSidebar"] .stPopover .stButton > button,
section[data-testid="stSidebar"] [data-testid="stPopover"] .stButton > button {
    justify-content: flex-start !important;
}
.stPopover [data-testid="stPopoverBody"] .danger-btn > button {
    color: #d32f2f !important;
    border-color: #d32f2f !important;
}
.stPopover [data-testid="stPopoverBody"] .danger-btn > button:hover {
    background-color: #ffebee !important;
}
.teacher-mode-banner {
    background: linear-gradient(135deg, #e8f5e9, #c8e6c9);
    border: 1.5px solid #43a047;
    border-radius: 10px;
    padding: 10px 16px;
    margin-bottom: 10px;
    color: #1b5e20;
    font-size: 0.92rem;
}
/* ── Ensemble Vote winner card ── */
.ensemble-winner {
    background: linear-gradient(135deg, #fff8e1, #ffecb3);
    border: 2px solid #f9a825;
    border-radius: 12px;
    padding: 14px 20px;
    margin-bottom: 12px;
}
</style>
""", unsafe_allow_html=True)

# ── Model Pricing Table (From poe.com) ──────────────────────────────────────────────────────
MODEL_PRICING = {
    "gemini-3.1-pro":          {"input": 0.002   / 1000, "output": 0.012  / 1000},
    "gemini-3-flash":          {"input": 0.0005  / 1000, "output": 0.003  / 1000},
    "gpt-5.4":                 {"input": 0.0025  / 1000, "output": 0.014  / 1000},
    "gpt-5.4-mini":            {"input": 0.00075 / 1000, "output": 0.0045  / 1000},
    "qwen3.5-flash":           {"input": 0.00009  / 1000, "output": 0.00037 / 1000},
    "kimi-k2.5":               {"input": 0.00061  / 1000, "output": 0.003  / 1000},
}

# ── Subject Prompts ──────────────────────────────────────────────────────────
SUBJECT_PROMPTS = {
    "🔢 DSE Mathematics (Core)": """You are an AI learning companion specializing in HKDSE Mathematics (Compulsory Part).
Rules:
1. Show step-by-step solutions aligned with DSE marking scheme Method Marks
2. Use $...$ for inline math formulas and $$...$$ for display equations
3. Annotate each step with the relevant knowledge point, e.g. [Quadratic Equation]
4. Provide exam tips and common mistake warnings at the end
5. Reply in Traditional Chinese; attach English for technical terms
6. For geometry questions, clearly explain the application of geometric properties""",

    "📐 DSE Mathematics (M1/M2)": """You are an AI learning companion specializing in HKDSE Mathematics Extension (M1/M2).
Rules:
1. Show full step-by-step solutions including complete calculus/statistics derivations
2. Use $...$ for inline math formulas and $$...$$ for display equations
3. Label the corresponding M1/M2 syllabus knowledge point for each step
4. Provide exam tips and common mistake warnings at the end
5. Reply in Traditional Chinese; attach English for technical terms""",

    "📖 DSE English Language": """You are an AI learning companion specializing in HKDSE English Language.
Rules:
1. For writing tasks, provide structured feedback on Content, Language, and Organization
2. Highlight grammar errors and suggest corrections
3. Provide vocabulary alternatives to enhance expression
4. Give specific DSE marking criteria advice
5. Respond in English, with Chinese explanations for complex grammar rules""",

    "📝 DSE Chinese Language": """You are an AI learning companion specializing in HKDSE Chinese Language.
Rules:
1. For reading comprehension, analyse paragraph by paragraph and quote the original text
2. Essay corrections should address Content, Expression, and Structure
3. Classical Chinese questions require character-by-character explanation and translation
4. Reply in Traditional Chinese
5. Reference DSE marking criteria throughout""",

    "🧪 DSE Physics": """You are an AI learning companion specializing in HKDSE Physics.
Rules:
1. Show formula, substitution, and full calculation for every question
2. Use $...$ for physics formulas
3. Label SI units and physical quantity symbols
4. Illustrate physics concepts with real-life examples
5. Reply in Traditional Chinese; use standard English symbols for formulas and units""",

    "💻 IT & Programming": """You are an AI learning companion specializing in IT and programming education.
Rules:
1. Wrap code in the appropriate language code block (e.g. ```python)
2. Explain code logic line by line, especially for beginners
3. When pointing out errors, explain the cause and the fix
4. Provide optimization suggestions and best practices
5. Reply in Simplified Chinese; write code comments in English""",

    "🤖 General AI Assistant": """You are a general-purpose tutoring assistant. Identify questions in images when uploaded.
When outputting math formulas, use $...$ for inline and $$...$$ for display equations.
Structure your answers clearly with step-by-step explanations.""",
}

TEACHER_MODE_SYSTEM_ADDON = """

[TEACHER MODE ACTIVE — These rules override all previous rules]

You now operate using the Socratic teaching method. Your goal is to guide students
to reach the answer through independent thinking.

━━━ Rule 1: When first receiving a question ━━━
• Solve the problem internally (do NOT output the solution)
• Identify the core knowledge point(s) and tell the student only the topic area
• Give one entry-point hint, e.g. "You might start by thinking about..."
• Prompt the student to attempt an answer with 👇, e.g.:
  "What do you think the answer is? Please try first 👇"
• ⛔ NEVER reveal any solution steps or the final answer before the student responds

━━━ Rule 2: After the student submits an answer ━━━
✅ Case A — Answer is fully correct:
   Reply "✅ Correct! 🎉" then reward the student with the full worked solution,
   explaining the reason for each step.

❌ Case B — Answer is wrong or incomplete:
   • Use an encouraging tone (e.g. "You're on the right track! Think again about...")
   • Reveal only the hint for the very next smallest step; never give the final answer
   • Invite the student to try again

━━━ Rule 3: Continuous guidance ━━━
• Repeat Rule 2, advancing only one small step at a time
• If the student has failed to answer correctly after 5 consecutive attempts,
  you may provide the complete worked solution and summarise the learning points
• Maintain an encouraging and patient teaching tone throughout
"""


# ==========================================
# 2. Ensemble Judge Function  ── NEW ──
# ==========================================

def run_ensemble_judge(
    api_key: str,
    base_url: str,
    judge_model: str,
    question: str,
    subject: str,
    responses_data: list[dict],
) -> dict:
    """
    Use a selected model (e.g. Claude Sonnet via Poe API) as Ensemble Judge
    to score all model responses via OpenAI-compatible client.

    Each response is evaluated on 4 dimensions (0-10 each):
    correctness, reasoning, completeness, clarity
    """
    client = OpenAI(api_key=api_key, base_url=base_url)

    # Build responses block; cap each at 2000 chars to stay within context
    responses_block = ""
    for i, r in enumerate(responses_data, 1):
        preview = (
            r["response"][:2000] + "\n…（Excerpt）"
            if len(r["response"]) > 2000
            else r["response"]
        )
        responses_block += (
            f"\n{'─'*50}\n"
            f"【Model {i}】{r['model']}\n"
            f"{'─'*50}\n"
            f"{preview}\n"
        )

    judge_prompt = f"""You are a senior HKDSE {subject} Subject graders，Simultaneously serving as an AI systems researcher，
responsible for ensemble voting to evaluate the responses from multiple AI models.。

【Title】
{question}

【Each model answers】
{responses_block}

━━━ Scoring dimensions (integers 0–10 for each item) ━━━
1. correctness   — Are the answers and calculation processes correct?
2. reasoning     — Is the problem-solving approach clear and logically rigorous?
3. completeness  — Complete steps to determine if the DSE assessment criteria are met.
4. clarity       — Are the explanations clear, easy to understand, and valuable for teaching?

Please return the evaluation results in JSON format. （**Return only JSON, no other text.**）：

{{
    "model_scores": [
        {{
            "model": "Model name (must be exactly the same as above)",
            "correctness": Integer,
            "reasoning": Integer,
            "completeness": Integer,
            "clarity": Integer,
            "total": The sum of four items,
            "comment": "Comments of 50 characters or less"
        }}
    ],
    "winner": "Highest score model name",
    "runner_up": "Second model name (if there is only one model, fill in N/A)",
    "winner_reason": "Explain why the model won in 80 characters or less (in Traditional Chinese).",
    "consensus_answer": "A summary of the most likely correct answer (within 100 characters, in Traditional Chinese) based on all model responses.",
    "overall_analysis": "A comparative study that is valuable for the research of this paper (within 150 characters, in Traditional Chinese)."
}}"""

    # ── Use OpenAI-compatible client (works with Poe API, OpenRouter, etc.) ──
    response = client.chat.completions.create(
        model       = judge_model,
        messages    = [
            {
                "role":    "system",
                "content": (
                    "You are a strict JSON-only responder. "
                    "Output ONLY valid JSON with no markdown fences, "
                    "no explanation, no extra text."
                ),
            },
            {"role": "user", "content": judge_prompt},
        ],
        temperature = 0.1,   # Low temperature for consistent structured output
        stream      = False,
    )

    raw = response.choices[0].message.content.strip()

    # Strip markdown code fences if the model wraps the JSON anyway
    if raw.startswith("```"):
        raw = re.sub(r"```(?:json)?\n?", "", raw).rstrip("` \n").strip()

    return json.loads(raw)


# ==========================================
# 3. Utility Functions  (unchanged)
# ==========================================

def encode_image_to_base64(file):
    return base64.b64encode(file.getvalue()).decode("utf-8")


def format_math(text):
    return (
        text.replace("\\[", "$$")
            .replace("\\]", "$$")
            .replace("\\(", "$")
            .replace("\\)", "$")
    )


def calculate_cost(prompt_tokens, completion_tokens, model_name):
    if model_name in MODEL_PRICING:
        p = MODEL_PRICING[model_name]
        return prompt_tokens * p["input"] + completion_tokens * p["output"]
    return 0.0


def prune_context(messages, max_messages=10):
    if len(messages) > max_messages:
        return messages[-max_messages:]
    return messages


def clear_delete_checkboxes():
    for k in [k for k in st.session_state if k.startswith("del_msg_")]:
        del st.session_state[k]


def create_new_session():
    session_id = str(uuid.uuid4())[:8]
    st.session_state.sessions[session_id] = {
        "title":              "New Chat",
        "messages":           [],
        "created_at":         time.time(),
        "last_active":        time.time(),
        "last_uploaded_file": None,
    }
    st.session_state.current_session_id = session_id
    return session_id


def get_current_session():
    return st.session_state.sessions[st.session_state.current_session_id]


def auto_update_title(session_id):
    session  = st.session_state.sessions[session_id]
    messages = session["messages"]
    if session["title"] != "New Chat":
        return
    user_msgs = [m for m in messages if m["role"] == "user"]
    if not user_msgs:
        return
    first = user_msgs[0]["content"]
    if isinstance(first, str):
        raw = first
    elif isinstance(first, list):
        raw = next(
            (item["text"] for item in first
             if item["type"] == "text"
             and item["text"] not in [
                 "Image uploaded:",
                 "Please solve the question in the image with full steps.",
             ]),
            "Image Question",
        )
    else:
        raw = "New Chat"
    session["title"] = raw[:18] + ("..." if len(raw) > 18 else "")


def reset_common_state():
    st.session_state.delete_mode         = False
    st.session_state.confirm_delete      = False
    st.session_state.pending_attachment  = None
    st.session_state.renaming_session_id = None
    clear_delete_checkboxes()


def build_system_prompt(subject: str, teacher_mode: bool) -> str:
    base = SUBJECT_PROMPTS[subject]
    if teacher_mode:
        return base + TEACHER_MODE_SYSTEM_ADDON
    return base


# ==========================================
# 4. Session State Initialization  (unchanged)
# ==========================================

if "sessions" not in st.session_state:
    st.session_state.sessions = {}

if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = None

if not st.session_state.sessions:
    create_new_session()

if st.session_state.current_session_id not in st.session_state.sessions:
    remaining = sorted(
        st.session_state.sessions.items(),
        key=lambda x: x[1]["last_active"], reverse=True,
    )
    st.session_state.current_session_id = (
        remaining[0][0] if remaining else create_new_session()
    )

_global_defaults = {
    "total_cost":          0.0,
    "total_tokens":        0,
    "experiment_log":      [],
    "response_times":      [],
    "delete_mode":         False,
    "confirm_delete":      False,
    "pending_attachment":  None,
    "attach_key":          0,
    "input_counter":       0,
    "renaming_session_id": None,
    "teacher_mode":        True,
}
for _k, _v in _global_defaults.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

curr = get_current_session()


# ==========================================
# 5. Sidebar  (unchanged except minor note)
# ==========================================
with st.sidebar:

    st.header("⚙️ Model Settings")
    api_key  = st.text_input("API Key", type="password", key="api_key_input")
    base_url = st.text_input("API Base URL", value="https://api.poe.com/v1")

    model_list = [
        "gemini-3.1-pro", "gemini-3-flash", "gpt-5.4", "gpt-5.4-mini"
        ,"qwen3.5-flash", "kimi-k2.5",
    ]
    model_name  = st.selectbox("🤖 Select Model", model_list, index=0)
    subject     = st.selectbox("📚 Select Subject", list(SUBJECT_PROMPTS.keys()), index=0)
    max_context = st.slider(
        "📝 Max Context Length (messages)",
        min_value=4, max_value=20, value=10, step=2,
    )

    st.divider()

    st.header("🎓 Teaching Settings")
    teacher_mode = st.toggle(
        "Teacher Mode (Guided Learning)",
        value=st.session_state.teacher_mode,
        key="teacher_mode_toggle",
        help=(
            "When enabled, the AI will NOT give away the solution directly.\n"
            "⚠️ Only applies to Tab 1. Tab 2 is always unaffected."
        ),
    )
    st.session_state.teacher_mode = teacher_mode

    if teacher_mode:
        st.success("✅ Teacher Mode is ON", icon="🎓")
        st.caption("The AI will guide students to answer independently before revealing solutions.")
    else:
        st.caption("Teacher Mode is OFF. The AI will provide complete answers directly.")

    st.divider()

    st.header("💬 Conversations")
    if st.button("➕ New Chat", use_container_width=True, type="primary", key="new_chat_btn"):
        create_new_session()
        reset_common_state()
        st.rerun()

    st.caption(f"{len(st.session_state.sessions)} conversation(s)")

    sorted_sessions = sorted(
        st.session_state.sessions.items(),
        key=lambda x: x[1]["last_active"], reverse=True,
    )

    for sid, sdata in sorted_sessions:
        is_current  = (sid == st.session_state.current_session_id)
        is_renaming = (st.session_state.renaming_session_id == sid)

        if is_renaming:
            new_name = st.text_input(
                "New title",
                value            = sdata["title"],
                key              = f"rename_input_{sid}",
                label_visibility = "collapsed",
                placeholder      = "Enter new conversation title...",
            )
            rc, rx = st.columns(2)
            with rc:
                if st.button("✓ Confirm", key=f"rename_ok_{sid}",
                             use_container_width=True, type="primary"):
                    stripped = new_name.strip()
                    if stripped:
                        st.session_state.sessions[sid]["title"] = stripped
                    st.session_state.renaming_session_id = None
                    st.rerun()
            with rx:
                if st.button("✕ Cancel", key=f"rename_cancel_{sid}",
                             use_container_width=True):
                    st.session_state.renaming_session_id = None
                    st.rerun()
        else:
            title_col, menu_col = st.columns([5, 1])
            with title_col:
                if st.button(
                    sdata["title"],
                    key                 = f"sess_btn_{sid}",
                    use_container_width = True,
                    type                = "primary" if is_current else "secondary",
                    help                = (
                        f"Last active: {time.strftime('%m-%d %H:%M', time.localtime(sdata['last_active']))}"
                        f"  |  {len(sdata['messages'])} message(s)"
                    ),
                ):
                    if not is_current:
                        st.session_state.current_session_id = sid
                        reset_common_state()
                        st.rerun()
            with menu_col:
                with st.popover("⋮", use_container_width=True):
                    if st.button("✏️  Rename", key=f"menu_rename_{sid}",
                                 use_container_width=True):
                        st.session_state.renaming_session_id = sid
                        st.rerun()
                    st.divider()
                    st.markdown('<div class="danger-btn">', unsafe_allow_html=True)
                    if st.button("🗑️  Delete Chat", key=f"menu_delete_{sid}",
                                 use_container_width=True):
                        del st.session_state.sessions[sid]
                        if is_current:
                            remaining = sorted(
                                st.session_state.sessions.items(),
                                key=lambda x: x[1]["last_active"], reverse=True,
                            )
                            if remaining:
                                st.session_state.current_session_id = remaining[0][0]
                            else:
                                create_new_session()
                        reset_common_state()
                        st.rerun()
                    st.markdown('</div>', unsafe_allow_html=True)

    st.divider()

    st.header("📊 Experiment Stats")
    _c1, _c2 = st.columns(2)
    with _c1:
        st.metric("Total Tokens", f"{st.session_state.total_tokens:,}")
        _rounds = len([m for m in curr["messages"] if m["role"] == "user"])
        st.metric("Current Rounds", _rounds)
    with _c2:
        st.metric("Total Cost", f"${st.session_state.total_cost:.4f}")
        _avg = (
            sum(st.session_state.response_times) / len(st.session_state.response_times)
            if st.session_state.response_times else 0
        )
        st.metric("Avg Response", f"{_avg:.1f}s")

    st.divider()

    st.header("💾 Export Data")
    if st.session_state.experiment_log:
        _df_exp = pd.DataFrame(st.session_state.experiment_log)
        _csv    = _df_exp.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            label     = "📥 Download CSV",
            data      = _csv,
            file_name = f"ai_experiment_{int(time.time())}.csv",
            mime      = "text/csv",
        )
        st.caption(f"{len(st.session_state.experiment_log)} record(s)")
    else:
        st.caption("No data yet. Start chatting to generate experiment records.")

    if st.button("🗑️ Clear Current Chat"):
        curr["messages"]    = []
        curr["title"]       = "New Chat"
        curr["last_active"] = time.time()
        st.session_state.input_counter += 1
        reset_common_state()
        st.rerun()


# ==========================================
# 6. Main Interface
# ==========================================
curr = get_current_session()

st.title("🎓 AI Learning Companion")
st.caption(f"📂 Current Chat: **{curr['title']}**")

tab1, tab2 = st.tabs(["💬 Single-Model Chat", "⚖️ Multi-Model Comparison (Experiment)"])


# ============================================================
# TAB 1: Single-Model Chat  (completely unchanged)
# ============================================================
with tab1:

    info_col, edit_btn_col = st.columns([5, 1])
    with info_col:
        mode_label = "  |  🎓 **Teacher Mode ON**" if st.session_state.teacher_mode else ""
        st.caption(
            f"Subject: **{subject}**  |  Model: **{model_name}**  |  "
            f"Context: last **{max_context}** messages{mode_label}"
        )
    with edit_btn_col:
        if not st.session_state.delete_mode:
            if st.button("✏️ Edit Chat", use_container_width=True):
                st.session_state.delete_mode    = True
                st.session_state.confirm_delete = False
                clear_delete_checkboxes()
                st.rerun()

    if st.session_state.teacher_mode:
        st.markdown(
            """<div class="teacher-mode-banner">
            🎓 <strong>Teacher Mode is ON</strong>: The AI will NOT give the solution directly.
            Please attempt the question first — the full worked solution will be revealed once
            you answer correctly.</div>""",
            unsafe_allow_html=True,
        )

    if st.session_state.delete_mode:
        selected_indices = [
            i for i in range(len(curr["messages"]))
            if st.session_state.get(f"del_msg_{i}", False)
        ]
        op1, op2, op3, _ = st.columns([2, 2, 2, 4])
        with op1:
            if selected_indices:
                st.info(f"**{len(selected_indices)}** selected", icon="✅")
            else:
                st.caption("👆 Check messages to delete")
        with op2:
            if st.button(
                f"🗑️ Delete ({len(selected_indices)})",
                type             = "primary",
                disabled         = (len(selected_indices) == 0),
                use_container_width = True,
            ):
                st.session_state.confirm_delete = True
                st.rerun()
        with op3:
            if st.button("✕ Cancel Edit", use_container_width=True):
                st.session_state.delete_mode    = False
                st.session_state.confirm_delete = False
                clear_delete_checkboxes()
                st.rerun()

        if st.session_state.confirm_delete and selected_indices:
            with st.container(border=True):
                st.warning(
                    f"⚠️ You are about to delete **{len(selected_indices)}** message(s). "
                    "This action **cannot be undone**. Continue?"
                )
                cc, cx, _ = st.columns([1.5, 1.5, 5])
                with cc:
                    if st.button("✅ Confirm Delete", type="primary",
                                 use_container_width=True, key="btn_confirm_del"):
                        curr["messages"] = [
                            m for i, m in enumerate(curr["messages"])
                            if i not in selected_indices
                        ]
                        st.session_state.delete_mode    = False
                        st.session_state.confirm_delete = False
                        clear_delete_checkboxes()
                        st.rerun()
                with cx:
                    if st.button("❌ Cancel", use_container_width=True, key="btn_cancel_del"):
                        st.session_state.confirm_delete = False
                        st.rerun()
        st.divider()

    def render_msg_content(msg):
        if "display_image" in msg:
            st.image(msg["display_image"], width=300)
        if isinstance(msg["content"], list):
            for item in msg["content"]:
                if item["type"] == "text":
                    _t = item["text"]
                    if _t and _t not in [
                        "Image uploaded:",
                        "Please solve the question in the image with full steps.",
                    ]:
                        st.markdown(format_math(_t))
        elif isinstance(msg["content"], str):
            st.markdown(format_math(msg["content"]))
        if msg["role"] == "assistant" and "stats" in msg:
            st.caption(msg["stats"])

    for i, msg in enumerate(curr["messages"]):
        if st.session_state.delete_mode:
            chk_col, msg_col = st.columns([1, 20])
            with chk_col:
                st.write("")
                st.checkbox(
                    label            = f"#{i+1}",
                    key              = f"del_msg_{i}",
                    label_visibility = "collapsed",
                )
            with msg_col:
                with st.chat_message(msg["role"]):
                    render_msg_content(msg)
        else:
            with st.chat_message(msg["role"]):
                render_msg_content(msg)

    if not st.session_state.delete_mode:
        with st.container(border=True):
            tb = st.columns([1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 14])
            _up_img = _up_pdf = None
            with tb[0]:
                with st.popover("🖼️", help="Upload image", use_container_width=True):
                    _up_img = st.file_uploader(
                        "Image", type=["png", "jpg", "jpeg"],
                        label_visibility = "collapsed",
                        key              = f"pop_img_{st.session_state.attach_key}",
                    )
            with tb[1]:
                with st.popover("📄", help="Upload PDF", use_container_width=True):
                    _up_pdf = st.file_uploader(
                        "PDF", type=["pdf"],
                        label_visibility = "collapsed",
                        key              = f"pop_pdf_{st.session_state.attach_key}",
                    )
            with tb[2]:
                st.button("🔗", disabled=True, help="Link (coming soon)",
                          use_container_width=True, key="tb_link")
            with tb[3]:
                st.button("🌐", disabled=True, help="Web search (coming soon)",
                          use_container_width=True, key="tb_web")
            with tb[4]:
                st.button("📋", disabled=True, help="Templates (coming soon)",
                          use_container_width=True, key="tb_tpl")
            with tb[5]:
                st.button("🗂️", disabled=True, help="More (coming soon)",
                          use_container_width=True, key="tb_more")

            _new_up = _up_img or _up_pdf
            if _new_up:
                _pend = st.session_state.pending_attachment
                if _pend is None or _pend.get("name") != _new_up.name:
                    st.session_state.pending_attachment = {
                        "data": _new_up.getvalue(),
                        "type": _new_up.type,
                        "name": _new_up.name,
                    }
                    st.session_state.attach_key += 1

            if st.session_state.pending_attachment:
                _att = st.session_state.pending_attachment
                _pc, _xc = st.columns([15, 1])
                with _pc:
                    if _att["type"].startswith("image"):
                        st.image(_att["data"], width=220)
                    else:
                        st.info(f"📎 **{_att['name']}**  ready", icon="📄")
                with _xc:
                    st.write("")
                    if st.button("✕", key="del_attach_inline"):
                        st.session_state.pending_attachment = None
                        st.session_state.attach_key += 1
                        st.rerun()

            _placeholder = (
                "Enter your answer or ask a new question..."
                if st.session_state.teacher_mode
                else "Type your question here..."
            )
            _user_text = st.text_area(
                "message",
                placeholder      = _placeholder,
                label_visibility = "collapsed",
                height           = 100,
                key              = f"chat_ta_{st.session_state.input_counter}",
            )
            _hint_col, _send_col = st.columns([16, 1])
            with _hint_col:
                if _user_text:
                    st.caption(f"📝 {len(_user_text)} characters")
            with _send_col:
                _do_send = st.button("↑", type="primary",
                                     use_container_width=True, key="do_send_btn")

        if _do_send:
            if not api_key:
                st.toast("⚠️ Please enter your API Key in the sidebar first!")
                st.stop()

            _att = st.session_state.pending_attachment
            _txt = _user_text.strip()

            if not _txt and not _att:
                st.toast("⚠️ Please type a question or attach a file!")
                st.stop()

            if _att and _att["type"].startswith("image"):
                _b64 = base64.b64encode(_att["data"]).decode()
                _user_msg = {
                    "role": "user",
                    "content": [
                        {"type": "text",
                         "text": _txt if _txt else "Please solve the question in the image with full steps."},
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/jpeg;base64,{_b64}"}},
                    ],
                    "display_image": _att["data"],
                }
            elif _att and "pdf" in _att["type"]:
                _user_msg = {
                    "role": "user",
                    "content": (
                        _txt + f"\n\n(File uploaded: {_att['name']})"
                        if _txt else f"Please analyse the uploaded file: {_att['name']}"
                    ),
                }
            else:
                _user_msg = {"role": "user", "content": _txt}

            curr["messages"].append(_user_msg)
            curr["last_active"] = time.time()
            auto_update_title(st.session_state.current_session_id)

            st.session_state.pending_attachment = None
            st.session_state.attach_key        += 1

            with st.chat_message("user"):
                render_msg_content(_user_msg)

            _system_prompt = build_system_prompt(subject, st.session_state.teacher_mode)

            client = OpenAI(api_key=api_key, base_url=base_url)
            with st.chat_message("assistant"):
                _ph           = st.empty()
                full_response = ""
                p_tok = c_tok = t_tok = 0
                start_time = time.time()

                try:
                    api_messages = [{"role": "system", "content": _system_prompt}]
                    pruned = prune_context(curr["messages"], max_context)
                    for m in pruned:
                        api_messages.append({"role": m["role"], "content": m["content"]})

                    stream = client.chat.completions.create(
                        model          = model_name,
                        messages       = api_messages,
                        stream         = True,
                        temperature    = 0.3,
                        stream_options = {"include_usage": True},
                    )

                    for chunk in stream:
                        if chunk.choices:
                            delta = chunk.choices[0].delta
                            if delta.content:
                                full_response += delta.content
                                _ph.markdown(format_math(full_response) + "▌")
                        if hasattr(chunk, "usage") and chunk.usage:
                            p_tok = chunk.usage.prompt_tokens
                            c_tok = chunk.usage.completion_tokens
                            t_tok = chunk.usage.total_tokens

                    _ph.markdown(format_math(full_response))

                    response_time = time.time() - start_time
                    cost          = calculate_cost(p_tok, c_tok, model_name)
                    word_count    = len(full_response)

                    st.session_state.total_cost   += cost
                    st.session_state.total_tokens += t_tok
                    st.session_state.response_times.append(response_time)

                    mode_tag   = "Teacher Mode" if st.session_state.teacher_mode else "Normal Mode"
                    stats_text = (
                        f"⏱️ {response_time:.2f}s  |  💰 ${cost:.6f}  |  "
                        f"🔢 {p_tok}+{c_tok}={t_tok} tokens  |  "
                        f"📝 {word_count} chars  |  🤖 {model_name}  |  🎓 {mode_tag}"
                    )
                    st.caption(stats_text)

                    st.session_state.experiment_log.append({
                        "Timestamp":        time.strftime("%Y-%m-%d %H:%M:%S"),
                        "Chat Title":       curr["title"],
                        "Test Type":        "Single-Model Chat",
                        "Teaching Mode":    mode_tag,
                        "Subject":          subject,
                        "Model":            model_name,
                        "Question":         _txt[:50] + ("..." if len(_txt) > 50 else ""),
                        "Input Tokens":     p_tok,
                        "Output Tokens":    c_tok,
                        "Total Tokens":     t_tok,
                        "Cost (USD)":       round(cost, 6),
                        "Response Time(s)": round(response_time, 2),
                        "Response Length":  word_count,
                        "Context Size":     len(pruned),
                    })

                    curr["messages"].append({
                        "role":    "assistant",
                        "content": full_response,
                        "stats":   stats_text,
                    })
                    curr["last_active"] = time.time()

                except Exception as e:
                    st.error(f"Error: {str(e)}")

            st.session_state.input_counter += 1
            st.rerun()


# ============================================================
# TAB 2: Multi-Model Comparison  ── MODIFIED ──
#   • 6 model slots (A–C required, D–F optional)
#   • Dynamic result display (up to 2 rows × 3 columns)
#   • Ensemble Voting via Claude Sonnet judge
# ============================================================
with tab2:
    st.subheader("⚖️ Multi-Model Side-by-Side Comparison")
    st.info(
        "📌 **Core Research Feature**: Send the exact same question to **up to 6 models** "
        "simultaneously. Compare answer quality, token usage, and response speed. "
        "Enable **Ensemble Voting** to let Claude Sonnet act as judge and pick the best answer. "
        "All results can be exported as CSV for dissertation data analysis."
    )

    # Teacher Mode isolation notice
    if st.session_state.teacher_mode:
        st.warning(
            "⚠️ Teacher Mode is enabled in the sidebar, but **this comparison test is unaffected**. "
            "All models will provide complete direct answers to ensure objective experiment data.",
            icon="🔬",
        )

    # ── Model Selection: 6 slots ─────────────────────────────────────────────
    st.markdown("**Select 3–6 models to compare（D、E、F is Optional）：**")

    # Sentinel value for "not selected" optional slots
    _NONE     = "（No Optional）"
    _opt_list = [_NONE] + model_list   # prepend the skip option

    # Row 1 — required
    _ca, _cb, _cc = st.columns(3)
    with _ca:
        model_a = st.selectbox("Model A ✳️", model_list, index=0, key="cmp_a")
    with _cb:
        model_b = st.selectbox("Model B ✳️", _opt_list, index=1, key="cmp_b")
    with _cc:
        model_c = st.selectbox("Model C ✳️", _opt_list, index=3, key="cmp_c")

    # Row 2 — optional
    _cd, _ce, _cf = st.columns(3)
    with _cd:
        model_d = st.selectbox("Model D （Optional）", _opt_list, index=0, key="cmp_d")
    with _ce:
        model_e = st.selectbox("Model E （Optional）", _opt_list, index=0, key="cmp_e")
    with _cf:
        model_f = st.selectbox("Model F （Optional）", _opt_list, index=0, key="cmp_f")

    # ── Subject / Image / Question ────────────────────────────────────────────
    cmp_subject = st.selectbox(
        "📚 Subject (for comparison)",
        list(SUBJECT_PROMPTS.keys()), index=0, key="cmp_subject",
    )
    cmp_image = st.file_uploader(
        "Upload question image (optional — sent to all models)",
        type=["png", "jpg", "jpeg"], key="cmp_image",
    )
    cmp_prompt = st.text_area(
        "📝 Enter test question (all models receive exactly the same content)",
        placeholder=(
            "E.g.: In △ABC, ∠A = 60°, BC = 5. "
            "Find the minimum value of AB + AC. Show full working."
        ),
        height=120, key="cmp_prompt",
    )

    # ── Ensemble Voting Settings ─────────────────────────────────────────────
    st.divider()
    with st.container(border=True):
        st.markdown("### 🗳️ Ensemble Voting Settings")
        st.caption(
            "Using a specified model as the judge, the evaluation is based on four dimensions，「correctness， reasoning, completeness, and clarity」"
            "The system scores all models, automatically selects the best answer, and generates an analytical report that can be cited in the paper.\n\n"
            "✅ **Use the same API Key and Base URL as above (Poe API compatible)**, no additional key required."
        )

        _ev_col, _ejm_col = st.columns([1, 2])
        with _ev_col:
            enable_ensemble = st.checkbox(
                "Enable Ensemble voting",
                value = True,
                help  = (
                    "The referee model uses the same API Key and Base URL as the sidebar.\n"
                    "Simply select the model to use as the referee on the right."
                ),
                key="enable_ensemble_chk",
            )
        with _ejm_col:
            if enable_ensemble:
                # Judge model list — includes Claude Sonnet variants common on Poe/aggregators
                _judge_model_options = [
                    "claude-sonnet-4.6",
                    "claude-sonnet-4.5",
                    "gpt-5.4",
                    "gpt-5.4-mini",
                    "gemini-3.1-pro",
                ] + model_list   # Append all standard models as fallback

                # Deduplicate while preserving order
                seen = set()
                _judge_model_options_dedup = []
                for _m in _judge_model_options:
                    if _m not in seen:
                        seen.add(_m)
                        _judge_model_options_dedup.append(_m)

                judge_model_selected = st.selectbox(
                    "Judgement Model",
                    _judge_model_options_dedup,
                    index = 0,
                    help  = (
                        "Select the judge model to evaluate the responses of other models.\n"
                        "It is recommended to use a more powerful model.（such as Claude Sonnet）。\n"
                        "The model name must be consistent with your API platform. "
                    ),
                    key="judge_model_sel",
                )

                # Allow custom model name override
                judge_model_custom = st.text_input(
                    "Customize the referee model name (leave blank to use the selection above).",
                    placeholder = "Example：claude-sonnet-4-6 (Subject to the actual name supported by the platform)",
                    key         = "judge_model_custom",
                )

                # Final judge model: custom input takes priority if filled
                judge_model = (
                    judge_model_custom.strip()
                    if judge_model_custom.strip()
                    else judge_model_selected
                )

                st.info(
                    f"✅ Use **`{judge_model}`** as a judge\n\n"
                    f"API Base URL：`{base_url}`",
                    icon="⚖️",
                )
            else:
                judge_model = ""

    # ── Run Button ────────────────────────────────────────────────────────────
    run_compare = st.button(
        "🚀 Run Comparison Test", type="primary", use_container_width=True
    )

    # ── Main Execution Block ──────────────────────────────────────────────────
    if run_compare:
        if not api_key:
            st.error("⚠️ Please enter your API Key in the sidebar first!")
            st.stop()
        if not cmp_prompt and not cmp_image:
            st.error("⚠️ Please enter a question or upload an image!")
            st.stop()

        # Build the final list of models to query (skip "（不選擇）" slots)
        selected_models = [
            m for m in [model_a, model_b, model_c, model_d, model_e, model_f]
            if m != _NONE
        ]

        # Build user content once — shared across all model calls
        if cmp_image:
            b64_cmp = encode_image_to_base64(cmp_image)
            user_content = [
                {
                    "type": "text",
                    "text": cmp_prompt if cmp_prompt
                            else "Please solve the question in the image with full steps.",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64_cmp}"},
                },
            ]
        else:
            user_content = cmp_prompt

        compare_results    = []   # raw metrics for the summary table
        responses_for_judge = []  # (model, response) pairs for the ensemble judge

        st.markdown(f"### 📊 Model Responses — {len(selected_models)} models")

        # ── Query models in batches of 3 (one row per batch) ──────────────────
        for batch_start in range(0, len(selected_models), 3):
            batch       = selected_models[batch_start : batch_start + 3]
            result_cols = st.columns(3)          # always 3 columns; unused ones stay blank

            for col_idx, mdl in enumerate(batch):
                with result_cols[col_idx]:
                    st.markdown(f"#### 🤖 `{mdl}`")
                    resp_ph  = st.empty()        # streaming text placeholder
                    stats_ph = st.empty()        # final stats badge placeholder
                    full_resp = ""
                    p_tok = c_tok = t_tok = 0
                    t0 = time.time()

                    try:
                        cli = OpenAI(api_key=api_key, base_url=base_url)
                        # ★ Tab 2 uses SUBJECT_PROMPTS directly —
                        #   Teacher Mode add-on is intentionally NOT injected
                        msgs = [
                            {"role": "system",
                             "content": SUBJECT_PROMPTS[cmp_subject]},
                            {"role": "user",
                             "content": user_content},
                        ]
                        stream = cli.chat.completions.create(
                            model          = mdl,
                            messages       = msgs,
                            stream         = True,
                            temperature    = 0.3,
                            stream_options = {"include_usage": True},
                        )
                        for chunk in stream:
                            if chunk.choices:
                                delta = chunk.choices[0].delta
                                if delta.content:
                                    full_resp += delta.content
                                    resp_ph.markdown(format_math(full_resp) + "▌")
                            if hasattr(chunk, "usage") and chunk.usage:
                                p_tok = chunk.usage.prompt_tokens
                                c_tok = chunk.usage.completion_tokens
                                t_tok = chunk.usage.total_tokens

                        resp_ph.markdown(format_math(full_resp))
                        elapsed = time.time() - t0
                        cost    = calculate_cost(p_tok, c_tok, mdl)

                        st.session_state.total_cost   += cost
                        st.session_state.total_tokens += t_tok
                        st.session_state.response_times.append(elapsed)

                        stats_ph.success(
                            f"⏱️ **{elapsed:.2f}s**  |  💰 **${cost:.6f}**  |  "
                            f"🔢 **{p_tok}+{c_tok}={t_tok}** tokens  |  "
                            f"📝 **{len(full_resp)}** chars"
                        )

                        compare_results.append({
                            "Model":            mdl,
                            "Response Time(s)": round(elapsed, 2),
                            "Input Tokens":     p_tok,
                            "Output Tokens":    c_tok,
                            "Total Tokens":     t_tok,
                            "Cost (USD)":       round(cost, 6),
                            "Response Length":  len(full_resp),
                        })
                        responses_for_judge.append({
                            "model":    mdl,
                            "response": full_resp,
                        })
                        st.session_state.experiment_log.append({
                            "Timestamp":        time.strftime("%Y-%m-%d %H:%M:%S"),
                            "Chat Title":       "Multi-Model Comparison",
                            "Test Type":        "Multi-Model Comparison",
                            "Teaching Mode":    "N/A (Research Experiment)",
                            "Subject":          cmp_subject,
                            "Model":            mdl,
                            "Question":         cmp_prompt[:50] + ("..." if len(cmp_prompt) > 50 else ""),
                            "Input Tokens":     p_tok,
                            "Output Tokens":    c_tok,
                            "Total Tokens":     t_tok,
                            "Cost (USD)":       round(cost, 6),
                            "Response Time(s)": round(elapsed, 2),
                            "Response Length":  len(full_resp),
                            "Context Size":     1,
                        })

                    except Exception as e:
                        resp_ph.error(f"❌ API call failed: {str(e)}")
                        compare_results.append({
                            "Model":            mdl,
                            "Response Time(s)": -1,
                            "Input Tokens":     0,
                            "Output Tokens":    0,
                            "Total Tokens":     0,
                            "Cost (USD)":       0,
                            "Response Length":  0,
                        })

        # ── Performance Summary Table ─────────────────────────────────────────
        if compare_results:
            st.divider()
            st.subheader("📊 Comparison Summary（dissertation data table）")
            st.dataframe(pd.DataFrame(compare_results), use_container_width=True)

            valid = [r for r in compare_results if r["Response Time(s)"] > 0]
            if valid:
                fastest    = min(valid, key=lambda x: x["Response Time(s)"])
                cheapest   = min(valid, key=lambda x: x["Cost (USD)"])
                most_words = max(valid, key=lambda x: x["Response Length"])

                st.markdown("**🏆 Auto-Analysis Results:**")
                _r1, _r2, _r3 = st.columns(3)
                with _r1:
                    st.success(
                        f"⚡ Fastest Response\n\n"
                        f"**{fastest['Model']}**\n\n"
                        f"`{fastest['Response Time(s)']}s`"
                    )
                with _r2:
                    st.info(
                        f"💰 Lowest Cost\n\n"
                        f"**{cheapest['Model']}**\n\n"
                        f"`${cheapest['Cost (USD)']:.6f}`"
                    )
                with _r3:
                    st.warning(
                        f"📝 Most Detailed\n\n"
                        f"**{most_words['Model']}**\n\n"
                        f"`{most_words['Response Length']} chars`"
                    )

        # ── Ensemble Voting Block ─────────────────────────────────────────────
 # ── Ensemble Voting Block ─────────────────────────────────────────────
        if enable_ensemble and responses_for_judge:
            st.divider()
            st.subheader("🗳️ Ensemble Voting results (judges' scores)")

            if not judge_model:
                st.warning("⚠️ Please select or enter the referee model name above and then run the test again.")

            else:
                with st.spinner(f"⚖️ `{judge_model}` Judges are scoring, please wait.…"):
                    try:
                        vote = run_ensemble_judge(
                            api_key        = api_key,        # ✅  Key
                            base_url       = base_url,       # ✅  URL
                            judge_model    = judge_model,
                            question       = cmp_prompt or "（Image title, no text）",
                            subject        = cmp_subject,
                            responses_data = responses_for_judge,
                        )

                        # ① Winner announcement banner
                        _medals = ["🥇", "🥈", "🥉"] + ["🏅"] * 10
                        sorted_scores = sorted(
                            vote["model_scores"],
                            key=lambda x: x["total"], reverse=True,
                        )
                        st.markdown(
                            f"""<div class="ensemble-winner">
                            <h4>🏆 Ensemble Voting champion：{vote['winner']}</h4>
                            <p>🥈 Runner-up：{vote.get('runner_up', 'N/A')}</p>
                            <p>💬 {vote['winner_reason']}</p>
                            </div>""",
                            unsafe_allow_html=True,
                        )

                        # ② Per-model score metrics
                        st.markdown("#### 📊 Total score of each model")
                        _metric_cols = st.columns(len(sorted_scores))
                        for i, (col, ms) in enumerate(zip(_metric_cols, sorted_scores)):
                            with col:
                                st.metric(
                                    label = f"{_medals[i]} {ms['model']}",
                                    value = f"{ms['total']} / 40",
                                    delta = f"Correctness {ms['correctness']}/10",
                                )

                        # ③ Full scoring breakdown table
                        st.markdown("#### 📋 Four-dimensional scoring details")
                        _scores_df = pd.DataFrame(sorted_scores)[
                            ["model", "correctness", "reasoning",
                             "completeness", "clarity", "total", "comment"]
                        ].rename(columns={
                            "model":        "Model",
                            "correctness":  "correctness /10",
                            "reasoning":    "reasoning /10",
                            "completeness": "completeness /10",
                            "clarity":      "clarity /10",
                            "total":        "total /40",
                            "comment":      "comment",
                        })
                        _scores_df.index = range(1, len(_scores_df) + 1)
                        st.dataframe(_scores_df, use_container_width=True)

                        # ④ Consensus answer
                        st.markdown("#### 💡 Ensemble Comprehensive answer")
                        st.info(vote.get("consensus_answer", "(The referee did not provide this information.)"), icon="💡")

                        # ⑤ Overall analysis for dissertation
                        st.markdown("#### 📝 Overall comparative analysis (paper reference)")
                        st.write(vote.get("overall_analysis", "(The referee did not provide this information.)"))

                        # ⑥ Individual comments (collapsible)
                        with st.expander("🔍 Individual comments for each model (click to expand)"):
                            for ms in sorted_scores:
                                st.markdown(
                                    f"**{ms['model']}** — Total Score **{ms['total']}/40**"
                                )
                                st.caption(ms.get("comment", "（No comments）"))
                                st.markdown("---")

                        # ⑦ Log to experiment_log
                        st.session_state.experiment_log.append({
                            "Timestamp":        time.strftime("%Y-%m-%d %H:%M:%S"),
                            "Chat Title":       "Ensemble Vote",
                            "Test Type":        "Ensemble Voting",
                            "Teaching Mode":    "N/A",
                            "Subject":          cmp_subject,
                            "Model":            f"{judge_model} (judge) → Winner: {vote['winner']}",
                            "Question":         cmp_prompt[:50] + ("..." if len(cmp_prompt) > 50 else ""),
                            "Input Tokens":     0,
                            "Output Tokens":    0,
                            "Total Tokens":     0,
                            "Cost (USD)":       0,
                            "Response Time(s)": 0,
                            "Response Length":  len(str(vote)),
                            "Context Size":     len(responses_for_judge),
                        })

                    except json.JSONDecodeError as je:
                        st.error(
                            f"❌ The referee responded that the JSON parsing failed. {je}\n\n"
                            "Recommendation: Try using a more capable referee model, or rerun the test. "
                        )
                    except Exception as e:
                        st.error(f"❌ Ensemble Voting failed: {e}")

        st.caption(
            "💡 All data above has been automatically logged. "
            "Click **Download CSV** in the sidebar to export for analysis."
        )