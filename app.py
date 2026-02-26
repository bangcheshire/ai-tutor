import streamlit as st
import base64
from openai import OpenAI

# ==========================================
# 1. 基础配置与工具
# ==========================================

st.set_page_config(page_title="AI 学习伴侣", page_icon="🤖", layout="wide")

def encode_image_to_base64(file):
    """将上传的文件转换为 Base64 格式"""
    return base64.b64encode(file.getvalue()).decode('utf-8')

def format_math(text):
    """格式化数学公式，将 LaTeX 转为 Streamlit 可读格式"""
    return text.replace("\\[", "$$").replace("\\]", "$$").replace("\\(", "$").replace("\\)", "$")

# 初始化会话状态
if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_uploaded_file" not in st.session_state:
    st.session_state.last_uploaded_file = None

# ==========================================
# 2. 侧边栏：设置与文件上传
# ==========================================
with st.sidebar:
    st.header("⚙️ 大脑设置")
    api_key = st.text_input("API Key", type="password", key="api_key_input")
    base_url = st.text_input("API Base URL", value="https://api.openai.com/v1")
    
    # 【本次修改核心】：使用 st.selectbox 替换 st.text_input
    model_list = [
        "gemini-3.1-pro-preview", 
        "gpt-5.2", 
        "grok-4.1", 
        "gemini-3-flash-preview", 
        "doubao-pro-32k",
        "deepseek-v3-2-exp",
        "gpt-5.2-pro-2025-12-11",
        "doubao-pro-128k",
        "deepseek-reasoner-164k"
    ]
    model_name = st.selectbox("🤖 选择模型 (Model)", model_list, index=0)
    
    st.divider()
    
    st.header("📎 上传附件")
    uploaded_file = st.file_uploader("上传题目图片/文件", type=['png', 'jpg', 'jpeg', 'pdf'])
    
    st.info("💡 提示：上传图片后，AI 会自动将其存入对话历史，你可以接着在右侧对话框里提问。")

    if st.button("🗑️ 清空对话历史"):
        st.session_state.messages = []
        st.session_state.last_uploaded_file = None
        st.rerun()

# ==========================================
# 3. 处理文件上传逻辑
# ==========================================
if uploaded_file and uploaded_file.file_id != st.session_state.last_uploaded_file:
    st.session_state.last_uploaded_file = uploaded_file.file_id
    
    if uploaded_file.type in ['image/png', 'image/jpeg', 'image/jpg']:
        base64_img = encode_image_to_base64(uploaded_file)
        
        img_message = {
            "role": "user",
            "content": [
                {"type": "text", "text": "我上传了一张图片："},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}
                }
            ],
            "display_image": uploaded_file 
        }
        st.session_state.messages.append(img_message)
        
    elif "pdf" in uploaded_file.type:
        st.warning("目前 PDF 仅支持作为文件传输，建议截图后上传以获得最佳识图效果。")

# ==========================================
# 4. 主聊天界面
# ==========================================
st.title("💬 AI 学习助手")

# --- 核心循环：渲染历史消息 ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if "display_image" in msg:
            st.image(msg["display_image"], width=300)
        
        if isinstance(msg["content"], list):
            for item in msg["content"]:
                if item["type"] == "text":
                    st.markdown(format_math(item["text"]))
        else:
            st.markdown(format_math(msg["content"]))
            
        if msg["role"] == "assistant" and "stats" in msg:
            st.caption(msg["stats"])

# --- 底部输入框 ---
if prompt := st.chat_input("输入你的问题（例如：这道题怎么做？/ 详细解释下第二步）"):
    if not api_key:
        st.toast("⚠️ 请先在侧边栏输入 API Key！")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    client = OpenAI(api_key=api_key, base_url=base_url)
    
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        usage_stats_text = ""
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0
        
        try:
            api_messages = []
            api_messages.append({
                "role": "system", 
                "content": "你是一位全科辅导老师。请识别图片题目。输出数学公式时，请尽量使用 $...$ 包裹行内公式，$$...$$ 包裹独立公式。"
            })
            
            for m in st.session_state.messages:
                clean_msg = {"role": m["role"], "content": m["content"]}
                api_messages.append(clean_msg)

            stream = client.chat.completions.create(
                model=model_name,
                messages=api_messages,
                stream=True,
                temperature=0.3,
                stream_options={"include_usage": True} 
            )
            
            for chunk in stream:
                if len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    if delta.content:
                        full_response += delta.content
                        message_placeholder.markdown(format_math(full_response) + "▌")
                
                if hasattr(chunk, 'usage') and chunk.usage is not None:
                    prompt_tokens = chunk.usage.prompt_tokens
                    completion_tokens = chunk.usage.completion_tokens
                    total_tokens = chunk.usage.total_tokens
            
            message_placeholder.markdown(format_math(full_response))
            
            word_count = len(full_response)
            
            if total_tokens > 0:
                usage_stats_text = f"word count: {word_count}, prompt tokens: {prompt_tokens}, completion tokens: {completion_tokens}, total tokens used: {total_tokens}, model: {model_name}"
            else:
                usage_stats_text = f"word count: {word_count}, tokens: N/A (API未返回), model: {model_name}"
            
            st.caption(usage_stats_text)
            
            st.session_state.messages.append({
                "role": "assistant", 
                "content": full_response,
                "stats": usage_stats_text
            })
            
        except Exception as e:
            st.error(f"出错啦: {str(e)}")