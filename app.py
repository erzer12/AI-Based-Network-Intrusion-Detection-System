import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score
from groq import Groq
import os
import joblib

# --- PAGE SETUP ---
st.set_page_config(page_title="AI-NIDS Student Project", layout="wide")

st.title("🛡️ AI-Based Network Intrusion Detection System")
st.markdown("""
**Student Project**: This system uses **Random Forest** to detect Network attacks and **Groq AI** to explain the packets.
""")

# --- CONFIGURATION ---
DATA_FILE = "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
MODEL_FILE = "trained_model.joblib"
FEATURE_NAMES = ['Flow Duration', 'Total Fwd Packets', 'Total Backward Packets', 
                 'Total Length of Fwd Packets', 'Fwd Packet Length Max', 
                 'Flow IAT Mean', 'Flow IAT Std', 'Flow Packets/s']

# --- SIDEBAR: SETTINGS ---
st.sidebar.header("1. Settings")
groq_api_key = st.sidebar.text_input("Groq API Key (starts with gsk_)", type="password")
st.sidebar.caption("[Get a free key here](https://console.groq.com/keys)")

st.sidebar.header("2. Model Training")

@st.cache_data
def load_data(filepath):
    """Load and preprocess the dataset."""
    try:
        df = pd.read_csv(filepath, nrows=15000)
        df.columns = df.columns.str.strip()
        # Fix: Use non-deprecated pandas pattern
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna()
        return df
    except FileNotFoundError:
        return None

def save_model(clf, accuracy, feature_names, X_test, y_test):
    """Save the trained model and metadata to disk."""
    model_data = {
        'model': clf,
        'accuracy': accuracy,
        'features': feature_names,
        'X_test': X_test,
        'y_test': y_test
    }
    joblib.dump(model_data, MODEL_FILE)

def load_model():
    """Load a previously trained model from disk."""
    if os.path.exists(MODEL_FILE):
        try:
            return joblib.load(MODEL_FILE)
        except Exception:
            return None
    return None

def train_model(df):
    """Train a Random Forest classifier on the dataset."""
    features = FEATURE_NAMES
    target = 'Label'
    
    missing_cols = [c for c in features if c not in df.columns]
    if missing_cols:
        st.error(f"Missing columns in CSV: {missing_cols}")
        return None, 0, [], None, None

    X = df[features]
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Improved: Increased n_estimators for better accuracy
    clf = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    
    score = accuracy_score(y_test, clf.predict(X_test))
    return clf, score, features, X_test, y_test

def plot_feature_importance(clf, feature_names):
    """Display a horizontal bar chart of feature importances using matplotlib."""
    importances = clf.feature_importances_
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=True)
    
    # Create matplotlib horizontal bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = sns.color_palette('viridis', len(importance_df))
    bars = ax.barh(importance_df['Feature'], importance_df['Importance'], color=colors)
    
    # Add value labels on bars
    for bar, val in zip(bars, importance_df['Importance']):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2, 
                f'{val:.3f}', va='center', fontsize=10)
    
    ax.set_xlabel('Importance Score', fontsize=12)
    ax.set_title('Feature Importance (Random Forest)', fontsize=14, fontweight='bold')
    ax.set_xlim(0, max(importances) * 1.15)  # Add space for labels
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

def plot_confusion_matrix(y_true, y_pred, labels):
    """Display a confusion matrix heatmap with detailed metrics."""
    # Sort labels for consistent display (BENIGN first if present)
    labels = sorted(labels, key=lambda x: (0 if x == "BENIGN" else 1, x))
    
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    
    # Create matplotlib heatmap
    st.write("**Confusion Matrix:**")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', ax=ax, 
                cbar_kws={'label': 'Count'}, linewidths=0.5)
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title('Confusion Matrix Heatmap', fontsize=14, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    
    # Calculate overall metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Display metrics in columns
    st.write("**Overall Metrics:**")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{accuracy:.2%}")
    m2.metric("Precision", f"{precision:.2%}")
    m3.metric("Recall", f"{recall:.2%}")
    m4.metric("F1-Score", f"{f1:.2%}")
    
    # Per-class metrics
    st.write("**Per-Class Performance:**")
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    
    # Build a clean dataframe for per-class metrics
    per_class_data = []
    for label in labels:
        if label in report:
            per_class_data.append({
                'Class': label,
                'Precision': f"{report[label]['precision']:.2%}",
                'Recall': f"{report[label]['recall']:.2%}",
                'F1-Score': f"{report[label]['f1-score']:.2%}",
                'Support': int(report[label]['support'])
            })
    
    per_class_df = pd.DataFrame(per_class_data)
    st.dataframe(per_class_df, use_container_width=True, hide_index=True)

# --- APP LOGIC ---
df = load_data(DATA_FILE)

if df is None:
    st.error(f"Error: File '{DATA_FILE}' not found. Please upload it to the Files tab.")
    st.stop()

st.sidebar.success(f"✅ Dataset Loaded: {len(df)} rows")

# Try to load a previously saved model
saved_model = load_model()
if saved_model and 'model' not in st.session_state:
    st.session_state['model'] = saved_model['model']
    st.session_state['features'] = saved_model['features']
    st.session_state['X_test'] = saved_model['X_test']
    st.session_state['y_test'] = saved_model['y_test']
    st.session_state['accuracy'] = saved_model['accuracy']
    st.sidebar.info(f"♻️ Loaded saved model (Accuracy: {saved_model['accuracy']:.2%})")

if st.sidebar.button("🚀 Train Model Now"):
    with st.spinner("Training model... This may take a moment."):
        clf, accuracy, feature_names, X_test, y_test = train_model(df)
        if clf:
            st.session_state['model'] = clf
            st.session_state['features'] = feature_names
            st.session_state['X_test'] = X_test 
            st.session_state['y_test'] = y_test
            st.session_state['accuracy'] = accuracy
            # Save model for persistence
            save_model(clf, accuracy, feature_names, X_test, y_test)
            st.sidebar.success(f"✅ Training Complete! Accuracy: {accuracy:.2%}")

# --- MODEL PERFORMANCE TAB ---
if 'model' in st.session_state:
    tab1, tab2 = st.tabs(["📊 Threat Analysis", "📈 Model Performance"])
    
    with tab2:
        st.header("Model Performance Dashboard")
        
        col_perf1, col_perf2 = st.columns(2)
        
        with col_perf1:
            st.subheader("🎯 Feature Importance")
            st.caption("Which packet fields matter most for detection?")
            plot_feature_importance(st.session_state['model'], st.session_state['features'])
        
        with col_perf2:
            st.subheader("📉 Confusion Matrix")
            st.caption("How accurate is the model at classifying traffic?")
            y_pred = st.session_state['model'].predict(st.session_state['X_test'])
            labels = st.session_state['y_test'].unique()
            plot_confusion_matrix(st.session_state['y_test'], y_pred, labels)
    
    with tab1:
        st.header("Threat Analysis Dashboard")
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎲 Simulation")
            st.info("Pick a random packet from the test data to simulate live traffic.")
            
            if st.button("🎲 Capture Random Packet"):
                random_idx = np.random.randint(0, len(st.session_state['X_test']))
                packet_data = st.session_state['X_test'].iloc[random_idx]
                actual_label = st.session_state['y_test'].iloc[random_idx]
                
                st.session_state['current_packet'] = packet_data
                st.session_state['actual_label'] = actual_label
                
        if 'current_packet' in st.session_state:
            packet = st.session_state['current_packet']
            
            with col1:
                st.write("**Packet Header Info:**")
                st.dataframe(packet, use_container_width=True)

            with col2:
                st.subheader("🤖 AI Detection Result")
                prediction = st.session_state['model'].predict([packet])[0]
                
                if prediction == "BENIGN":
                    st.success(f"✅ STATUS: **SAFE (BENIGN)**")
                else:
                    st.error(f"🚨 STATUS: **ATTACK DETECTED ({prediction})**")
                
                st.caption(f"Ground Truth Label: {st.session_state['actual_label']}")

                st.markdown("---")
                st.subheader("🧠 Ask AI Analyst (Groq)")
                
                # Enhanced: Added option for remediation steps
                include_remediation = st.checkbox("Include remediation steps", value=True)
                
                if st.button("💡 Generate Explanation"):
                    if not groq_api_key:
                        st.warning("⚠️ Please enter your Groq API Key in the sidebar first.")
                    else:
                        try:
                            client = Groq(api_key=groq_api_key)
                            
                            remediation_prompt = """
                            4. If it's an attack, provide 2-3 specific remediation steps (e.g., firewall rules, rate limiting).
                            """ if include_remediation else ""
                            
                            prompt = f"""
                            You are a cybersecurity analyst. 
                            A network packet was detected as: {prediction}.
                            
                            Packet Technical Details:
                            {packet.to_string()}
                            
                            Please explain:
                            1. Why these specific values (like Flow Duration or Packet Length) might indicate {prediction}.
                            2. If it is BENIGN, explain why it looks normal.
                            3. Keep the answer short and simple for a student.
                            {remediation_prompt}
                            """

                            with st.spinner("Groq is analyzing the packet..."):
                                completion = client.chat.completions.create(
                                    model="llama-3.3-70b-versatile",
                                    messages=[
                                        {"role": "user", "content": prompt}
                                    ],
                                    temperature=0.6,
                                )
                                st.info(completion.choices[0].message.content)
                                
                        except Exception as e:
                            st.error(f"API Error: {e}")
else:
    st.info("⏳ Waiting for model training. Click **'Train Model Now'** in the sidebar.")