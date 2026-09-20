import os
from pathlib import Path

import pandas as pd
import streamlit as st
from google import genai
from google.genai import types

# ============================================================
# 1. CẤU HÌNH
# ============================================================
st.set_page_config(
    page_title="Trợ lý ảo nhà hàng",
    page_icon="🍽️",
    layout="centered",
)

BASE_DIR = Path(__file__).resolve().parent
MENU_FILE = BASE_DIR / "menu.csv"

# ============================================================
# 2. API KEY
# ============================================================
api_key = st.secrets.get("GOOGLE_API_KEY")

st.title("🍽️ Trợ lý ảo nhà hàng")
st.caption("Chatbot sử dụng Gemini và dữ liệu từ menu.csv")

if not api_key:
    st.error(
        "Không tìm thấy GOOGLE_API_KEY. "
        "Hãy tạo file key.env cùng thư mục với app.py."
    )
    st.code("GOOGLE_API_KEY=YOUR_API_KEY")
    st.stop()

# ============================================================
# 3. ĐỌC MENU
# ============================================================
if not MENU_FILE.exists():
    st.error("Không tìm thấy menu.csv. Hãy đặt menu.csv cùng thư mục với app.py.")
    st.stop()

try:
    menu_df = pd.read_csv(MENU_FILE)
    menu_df.columns = [str(c).strip() for c in menu_df.columns]
except Exception as e:
    st.error(f"Không thể đọc menu.csv: {e}")
    st.stop()

if menu_df.empty:
    st.error("menu.csv đang rỗng.")
    st.stop()

menu_text = menu_df.to_string(index=False)

# ============================================================
# 4. TẠO GEMINI CLIENT
# ============================================================
try:
    client = genai.Client(api_key=api_key)
except Exception as e:
    st.error(f"Không thể tạo Gemini client: {e}")
    st.stop()

# Có thể đặt GEMINI_MODEL trong key.env.
# Ví dụ: GEMINI_MODEL=gemini-3.6-flash
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")

# ============================================================
# 5. SYSTEM INSTRUCTION
# ============================================================
SYSTEM_INSTRUCTION = f"""
Bạn là trợ lý ảo của một nhà hàng.

NHIỆM VỤ:
- Trả lời câu hỏi về thực đơn.
- Giới thiệu các món ăn có trong menu.
- Tư vấn món ăn dựa trên thông tin có trong menu.
- Trả lời lịch sự, thân thiện, ngắn gọn.

QUY TẮC:
1. Chỉ sử dụng thông tin có trong dữ liệu menu được cung cấp.
2. Không tự bịa giá, nguyên liệu, món ăn, khuyến mãi hoặc thông tin khác.
3. Nếu thông tin khách hỏi không có trong menu, nói rõ rằng nhà hàng
   chưa cung cấp thông tin đó.
4. Nếu khách hỏi món không có trong menu, nói rằng món đó hiện không có.
5. Có thể giao tiếp bằng tiếng Việt.
6. Khi đề xuất món, chỉ đề xuất những món có trong menu.

DỮ LIỆU MENU.CSV:
{menu_text}
"""

# ============================================================
# 6. SESSION STATE
# ============================================================
if "conversation_log" not in st.session_state:
    st.session_state.conversation_log = []

# ============================================================
# 7. SIDEBAR
# ============================================================
with st.sidebar:
    st.header("⚙️ Cài đặt")

    st.write(f"**Model:** `{MODEL_NAME}`")

    if st.button("🗑️ Xóa lịch sử", use_container_width=True):
        st.session_state.conversation_log = []
        st.rerun()

    with st.expander("📋 Xem menu"):
        st.dataframe(menu_df, use_container_width=True)

# ============================================================
# 8. HIỂN THỊ LỊCH SỬ
# ============================================================
for message in st.session_state.conversation_log:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# ============================================================
# 9. CHAT
# ============================================================
user_prompt = st.chat_input("Nhập câu hỏi, ví dụ: Menu có những món gì?")

if user_prompt:
    with st.chat_message("user"):
        st.write(user_prompt)

    st.session_state.conversation_log.append(
        {"role": "user", "content": user_prompt}
    )

    contents = []
    for message in st.session_state.conversation_log:
        contents.append(
            types.Content(
                role=message["role"],
                parts=[types.Part(text=message["content"])],
            )
        )

    with st.chat_message("assistant"):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION
                ),
            )

            answer = response.text or "Xin lỗi, tôi chưa tạo được câu trả lời."
            st.write(answer)

            st.session_state.conversation_log.append(
                {"role": "model", "content": answer}
            )

        except Exception as e:
            st.error(f"Lỗi khi gọi Gemini: {e}")
            st.info(
                "Nếu lỗi là 404 model, hãy kiểm tra model mà API key của bạn "
                "được phép sử dụng và đổi biến GEMINI_MODEL trong key.env."
            )
