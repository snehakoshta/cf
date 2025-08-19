import pickle
import numpy as np
import streamlit as st
import difflib
import sqlite3
from datetime import datetime

# ------------------ Load Models ------------------
with open('decision_tree_model_crop.pkl', 'rb') as file:
    rf_model_crop = pickle.load(file)

with open('decision_tree_model_fertilizer.pkl', 'rb') as file:
    rf_model_fertilizer = pickle.load(file)

# ------------------ Database Setup ------------------
conn = sqlite3.connect("orders.db", check_same_thread=False)
c = conn.cursor()

# Orders Table
c.execute("""
CREATE TABLE IF NOT EXISTS orders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    address TEXT,
    payment_mode TEXT,
    product TEXT,
    quantity TEXT,
    category TEXT,
    date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
""")

# Users Table
c.execute("""
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE,
    password TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
""")
conn.commit()

# ------------------ DB Functions ------------------
def save_order(name, address, payment_mode, product, quantity, category):
    c.execute("INSERT INTO orders (name, address, payment_mode, product, quantity, category) VALUES (?, ?, ?, ?, ?, ?)",
              (name, address, payment_mode, product, quantity, category))
    conn.commit()

def get_orders():
    c.execute("SELECT id, name, address, payment_mode, product, quantity, category, date FROM orders ORDER BY date DESC")
    return c.fetchall()

def delete_order(order_id):
    c.execute("DELETE FROM orders WHERE id = ?", (order_id,))
    conn.commit()

def clear_all_orders():
    c.execute("DELETE FROM orders")
    conn.commit()

def register_user(username, password):
    try:
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False

def login_user(username, password):
    c.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
    return c.fetchone()

# ------------------ Styles ------------------
def add_global_styles():
    st.markdown(
        """
        <style>
        .stApp {
            background: url("https://png.pngtree.com/thumb_back/fh260/background/20210302/pngtree-crop-green-rice-light-effect-wallpaper-image_571433.jpg") no-repeat center center fixed;
            background-size: cover;
        }
        .result-box {
            background: linear-gradient(135deg, #228B22, #32CD32);
            color: white;
            font-size: 18px;
            font-weight: bold;
            border-radius: 16px;
            padding: 18px;
            margin: 14px 0;
            box-shadow: 0px 4px 15px rgba(0,0,0,0.25);
        }
        .order-box {
            background: #fff;
            border-left: 6px solid #32CD32;
            padding: 12px;
            margin: 10px 0;
            border-radius: 10px;
            font-size: 15px;
            box-shadow: 0px 4px 12px rgba(0,0,0,0.15);
        }
        .chat-bubble { padding: 12px 16px; margin: 8px 0; border-radius: 14px; max-width: 75%; }
        .user-bubble { background: #d1f5ff; margin-left: auto; border: 1px solid #a3e4ff; }
        .bot-bubble { background: #e6ffe6; margin-right: auto; border: 1px solid #b3ffb3; }
        </style>
        """,
        unsafe_allow_html=True
    )

# ------------------ Recommendation & Chatbot ------------------
crop_descriptions = {
    "Wheat": "Wheat is a staple crop...",
    "Rice": "Rice thrives in clayey soils...",
    "Maize": "Maize requires well-drained loamy soil...",
    "Barley": "Barley is a hardy crop..."
}
fertilizer_descriptions = {
    "Urea": "Urea is rich in Nitrogen...",
    "DAP": "Di-ammonium Phosphate...",
    "MOP": "Muriate of Potash...",
    "Compost": "Organic compost improves soil fertility..."
}
knowledge_base = {
    "what is npk": "NPK stands for Nitrogen...",
    "best fertilizer for wheat": "Urea and DAP...",
    "best soil for rice": "Clayey soil...",
}

def recommend_crop(ph, humidity, N, P, K, temperature, rainfall):
    features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    prediction = rf_model_crop.predict(features)
    return prediction[0]

def recommend_fertilizer(temperature, humidity, moisture, soil_type, crop_type, N, P, K):
    soil_mapping = {'Loamy': 0, 'Sandy': 1, 'Clayey': 2}
    crop_mapping = {'Wheat': 0, 'Rice': 1, 'Maize': 2, 'Barley': 3}
    features = np.array([[temperature, humidity, moisture,
                          soil_mapping.get(soil_type, 0),
                          crop_mapping.get(crop_type, 0),
                          N, P, K]])
    prediction = rf_model_fertilizer.predict(features)
    return prediction[0]

def chatbot_response(user_input):
    import difflib
    user_input_lower = user_input.lower()
    best_match = difflib.get_close_matches(user_input_lower, knowledge_base.keys(), n=1, cutoff=0.5)
    if best_match:
        return knowledge_base[best_match[0]]
    return "Sorry, I don't have an answer for that."

# ------------------ Streamlit App ------------------
st.set_page_config(page_title="🌾 Crop & Fertilizer System", layout="wide")
add_global_styles()
st.title("🌾 Crop & Fertilizer Recommendation System")

# ------------------ LOGIN PAGE ------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""

if not st.session_state.logged_in:
    st.header("🔑 Login / Register")

    choice = st.radio("Choose Action", ["Login", "Register"])
    username = st.text_input("👤 Username")
    password = st.text_input("🔒 Password", type="password")

    if choice == "Login":
        if st.button("Login"):
            user = login_user(username, password)
            if user:
                st.session_state.logged_in = True
                st.session_state.username = username
                st.success(f"✅ Welcome back, {username}!")
                st.rerun()
            else:
                st.error("❌ Invalid username or password")
    else:
        if st.button("Register"):
            if register_user(username, password):
                st.success("✅ Registration successful! Please login.")
            else:
                st.error("⚠️ Username already exists!")

else:
    st.sidebar.success(f"👤 Logged in as {st.session_state.username}")
    if st.sidebar.button("🚪 Logout"):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.rerun()

    # After login → Show all other tabs
    tab1, tab2, tab3, tab4 = st.tabs(
        ["🌱 Crop & Fertilizer", "🤖 Chatbot", "🛍 Order Page", "📜 Order History"]
    )

    # ------------------ Tab1: Crop & Fertilizer ------------------
    with tab1:
        st.header("🌱 Enter Soil & Weather Details")
        col1, col2 = st.columns(2)
        with col1:
            ph = st.number_input("Soil pH", 0.0, 14.0, 6.5)
            humidity = st.number_input("Humidity (%)", 0.0, 100.0, 60.0)
            N = st.number_input("Nitrogen (N)", 0, 100, 50)
            P = st.number_input("Phosphorous (P)", 0, 100, 40)
            K = st.number_input("Potassium (K)", 0, 100, 40)
        with col2:
            temperature = st.number_input("Temperature (°C)", -10.0, 50.0, 25.0)
            moisture = st.number_input("Moisture (%)", 0.0, 100.0, 30.0)
            soil_type = st.selectbox("Soil Type", ["Loamy", "Sandy", "Clayey"])
            crop_type = st.selectbox("Crop Type", ["Wheat", "Rice", "Maize", "Barley"])

        if st.button("🌱 Get Recommendations"):
            crop_result = recommend_crop(ph, humidity, N, P, K, temperature, rainfall=200.0)
            fert_result = recommend_fertilizer(temperature, humidity, moisture, soil_type, crop_type, N, P, K)

            st.markdown(f"<div class='result-box'>🌾 Recommended Crop: {crop_result}<br><i>{crop_descriptions.get(crop_result, '')}</i></div>", unsafe_allow_html=True)
            st.markdown(f"<div class='result-box'>🌿 Recommended Fertilizer: {fert_result}<br><i>{fertilizer_descriptions.get(fert_result, '')}</i></div>", unsafe_allow_html=True)

    # ------------------ Tab2: Chatbot ------------------
    with tab2:
        st.header("🤖 Chatbot Assistant")
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        user_input = st.text_input("Ask me anything about crops or fertilizers:")
        if st.button("Send"):
            if user_input:
                response = chatbot_response(user_input)
                st.session_state.chat_history.append(("user", user_input))
                st.session_state.chat_history.append(("bot", response))
        for role, msg in st.session_state.chat_history:
            bubble_class = "user-bubble" if role == "user" else "bot-bubble"
            st.markdown(f'<div class="chat-bubble {bubble_class}">{msg}</div>', unsafe_allow_html=True)

    # ------------------ Tab3: Order Page ------------------
    with tab3:
        st.header("🛍 Order Crops, Seeds & Fertilizers")
        name = st.text_input("👤 Enter Your Name", value=st.session_state.username)
        address = st.text_area("🏠 Enter Delivery Address")
        payment_mode = st.selectbox("💳 Payment Mode", ["Cash on Delivery", "UPI", "Net Banking", "Card Payment"])
        category = st.radio("Choose Category:", ["🌾 Crops", "🌱 Seeds", "💊 Fertilizers"])

        if category == "🌾 Crops":
            crop = st.selectbox("Select Crop:", ["Wheat", "Rice", "Maize", "Barley"])
            qty = st.number_input("Quantity (Quintals)", 1, 100, 1)
            if st.button("🛒 Order Crop"):
                save_order(name, address, payment_mode, crop, f"{qty} Quintals", "Crop")
                st.success(f"✅ Order placed for {qty} Quintals of {crop}")

        elif category == "🌱 Seeds":
            seed = st.selectbox("Select Seed:", ["Wheat Seed", "Rice Seed", "Maize Seed", "Barley Seed"])
            qty = st.number_input("Quantity (Kg)", 1, 500, 10)
            if st.button("🛒 Order Seeds"):
                save_order(name, address, payment_mode, seed, f"{qty} Kg", "Seed")
                st.success(f"✅ Order placed for {qty} Kg of {seed}")

        else:
            fert = st.selectbox("Select Fertilizer:", ["Urea", "DAP", "MOP", "Compost"])
            qty = st.number_input("Quantity (Bags)", 1, 200, 1)
            if st.button("🛒 Order Fertilizer"):
                save_order(name, address, payment_mode, fert, f"{qty} Bags", "Fertilizer")
                st.success(f"✅ Order placed for {qty} Bags of {fert}")

    # ------------------ Tab4: Order History ------------------
    with tab4:
        st.header("📜 Your Order History")
        orders = get_orders()
        if not orders:
            st.info("No orders yet.")
        else:
            if st.button("🗑 Clear All Orders"):
                clear_all_orders()
                st.success("✅ All orders deleted.")
                st.rerun()
            for i, (oid, name, address, pm, product, qty, cat, date) in enumerate(orders, 1):
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.markdown(f"<div class='order-box'>📝 {i}. <b>{product}</b> - {qty} ({cat})<br>👤 {name} | 💳 {pm}<br>🏠 {address}<br>📅 {date}</div>", unsafe_allow_html=True)
                with col2:
                    if st.button("❌ Delete", key=f"del_{oid}"):
                        delete_order(oid)
                        st.warning(f"❌ Order {i} deleted")
                        st.error(
                            """
                            <div style="
                                background: white;
                                color: #d32f2f;
                                font-weight: bold;
                                border-radius: 10px;
                                padding: 12px;
                                animation: shake 0.4s;
                                border: 2px solid #d32f2f;
                                box-shadow: 0 2px 8px rgba(211,47,47,0.08);
                            ">
                                ❌ Invalid username or password
                            </div>
                            <style>
                            @keyframes shake {
                                0% { transform: translateX(0); }
                                20% { transform: translateX(-8px); }
                                40% { transform: translateX(8px); }
                                60% { transform: translateX(-8px); }
                                80% { transform: translateX(8px); }
                                100% { transform: translateX(0); }
                            }
                            </style>
                            """,
                            unsafe_allow_html=True
                        )