import os
import re
import joblib
import numpy as np
import pandas as pd
from urllib.parse import urlparse
from tensorflow.keras.models import load_model

# =====================================================================================
# 1. تحميل كافة النماذج والأدوات المحفوظة
# =====================================================================================
print("==================================================")
print("جاري تحميل المحرك الأمني المتكامل (IDS + MitM + Phishing)...")
print("==================================================")

# مسارات نماذج كشف التسلل الرئيسي
MLP_MODEL_PATH = "saved_models/mlp_model.keras"
CHAOTIC_MODEL_PATH = "saved_models/chaotic_model.keras"
SCALER_PATH = "saved_models/scaler.pkl"
FEATURES_PATH = "saved_models/feature_names.pkl"

# مسارات نموذج كشف الاحتيال (Phishing)
PHISHING_MODEL_PATH = "saved_models/phishing_model.pkl"
PHISHING_FEATURES_PATH = "saved_models/phishing_feature_names.pkl"

# تحميل النماذج
mlp_model = load_model(MLP_MODEL_PATH)
chaotic_model = load_model(CHAOTIC_MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
feature_names = joblib.load(FEATURES_PATH)

phishing_model = joblib.load(PHISHING_MODEL_PATH)
phishing_feature_names = joblib.load(PHISHING_FEATURES_PATH)

print("✅ تم تحميل محرك كشف التسلل ومحرك كشف الاحتيال بنجاح!\n")

# =====================================================================================
# 2. دالة حساب الميزات السلوكية وميزات MitM فورياً داخل IDS Engine (تم الإصلاح)
# =====================================================================================
def compute_live_behavior_and_mitm_features(df, time_window="60s"):
    """
    تقوم هذه الدالة بحساب الميزات السلوكية وميزات MitM فورياً على حركة المرور المدخلة
    """
    df = df.copy()
    
    # 1. معالجة التوقيت الزمني
    if 'Timestamp' in df.columns:
        df['dt'] = pd.to_datetime(df['Timestamp'], errors='coerce')
        if df['dt'].isna().sum() > 0:
            df['dt'] = df['dt'].bfill().ffill()
    else:
        df['dt'] = pd.date_range(start='2017-07-03 08:00:00', periods=len(df), freq='10ms')

    df = df.sort_values('dt').reset_index(drop=True)
    src_ip_col = 'Source IP' if 'Source IP' in df.columns else None
    
    if src_ip_col:
        grouper = df.groupby([src_ip_col, pd.Grouper(key='dt', freq=time_window)])
    else:
        grouper = df.groupby(pd.Grouper(key='dt', freq=time_window))

    flow_counts = grouper['dt'].transform('count')
    new_features = {}

    fwd_pkts = df['Total Fwd Packets'] if 'Total Fwd Packets' in df.columns else 1
    bwd_pkts = df['Total Backward Packets'] if 'Total Backward Packets' in df.columns else 0
    
    fwd_col_name = fwd_pkts.name if hasattr(fwd_pkts, 'name') and fwd_pkts.name in df.columns else None
    
    if fwd_col_name:
        new_features['Total_Packets_60s'] = grouper[fwd_col_name].transform('sum')
    else:
        new_features['Total_Packets_60s'] = fwd_pkts + bwd_pkts

    new_features['Packets_Per_Flow_60s'] = new_features['Total_Packets_60s'] / np.maximum(flow_counts, 1)
    
    dest_ip_col = 'Destination IP' if 'Destination IP' in df.columns else None
    dest_port_col = 'Destination Port' if 'Destination Port' in df.columns else None
    
    if dest_ip_col:
        new_features['Unique_Destination_IPs_60s'] = grouper[dest_ip_col].transform('nunique')
    else:
        new_features['Unique_Destination_IPs_60s'] = 1

    if dest_port_col:
        new_features['Unique_Destination_Ports_60s'] = grouper[dest_port_col].transform('nunique')
    else:
        new_features['Unique_Destination_Ports_60s'] = 1
    
    new_features['Destination_IP_Diversity_Ratio_60s'] = new_features['Unique_Destination_IPs_60s'] / np.maximum(flow_counts, 1)
    new_features['Port_Diversity_Ratio_60s'] = new_features['Unique_Destination_Ports_60s'] / np.maximum(flow_counts, 1)
    new_features['Connections_Per_Second_60s'] = flow_counts / 60.0
    new_features['Flows_Per_Source_60s'] = flow_counts
    
    ack_col_name = 'ACK Flag Count' if 'ACK Flag Count' in df.columns else None
    if ack_col_name:
        new_features['ACK_Count_60s'] = grouper[ack_col_name].transform('sum')
    else:
        new_features['ACK_Count_60s'] = 0
    
    # --- حساب ميزات MitM الفورية بشكل آمن ---
    if 'Bwd Packet Length Std' in df.columns:
        df['ack_dup_temp'] = (new_features['ACK_Count_60s'] * (df['Bwd Packet Length Std'] > 0)).astype(int)
        new_features['Duplicate_ACK_Count_60s'] = grouper['ack_dup_temp'].transform('sum')
        df.drop(columns=['ack_dup_temp'], inplace=True, errors='ignore')
    else:
        new_features['Duplicate_ACK_Count_60s'] = np.maximum(0, new_features['ACK_Count_60s'] - flow_counts)

    bwd_fwd_ratio = np.where(fwd_pkts > 0, bwd_pkts / np.maximum(fwd_pkts, 1), 0)
    df['arp_ratio_temp'] = bwd_fwd_ratio
    new_features['Arp_Reply_Ratio_60s'] = grouper['arp_ratio_temp'].transform('mean')
    df.drop(columns=['arp_ratio_temp'], inplace=True, errors='ignore')

    # دمج الميزات
    df_new = pd.DataFrame(new_features, index=df.index)
    df = pd.concat([df, df_new], axis=1)
    df.drop(columns=['dt'], inplace=True, errors='ignore')
    return df

# =====================================================================================
# 3. دالة فحص تدفقات الشبكة واستخراج التنبيهات (IDS Pipeline)
# =====================================================================================
def run_ids_detection(data_path):
    print(f"جاري قراءة وحساب الميزات السلوكية و MitM لملف: {data_path} ...")
    df_raw = pd.read_csv(data_path, low_memory=False, on_bad_lines='skip')
    df_raw.columns = df_raw.columns.str.strip()
    
    # 1. حساب الميزات السلوكية وميزات MitM فورياً
    df_processed = compute_live_behavior_and_mitm_features(df_raw)
    
    # 2. التأكد من توفر جميع الميزات التي تدرب عليها النموذج
    missing_cols = set(feature_names) - set(df_processed.columns)
    for col in missing_cols:
        df_processed[col] = 0
        
    X = df_processed[feature_names].copy()
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(0, inplace=True)
    
    # 3. التطبيع والتنبؤ المزدوج (MLP + Chaotic Network)
    X_scaled = scaler.transform(X)
    
    mlp_preds = (mlp_model.predict(X_scaled, batch_size=1024, verbose=0) > 0.5).astype(int).flatten()
    chaotic_preds = (chaotic_model.predict(X_scaled, batch_size=1024, verbose=0) > 0.5).astype(int).flatten()
    
    # منطق الفلترة الهجينة
    hybrid_preds = np.where((mlp_preds == 1) & (chaotic_preds == 1), 1, 0)
    
    total_flows = len(hybrid_preds)
    detected_attacks = np.sum(hybrid_preds)
    
    print("\n================ DETECTED TRAFFIC RESULTS ================")
    print(f"Total Evaluated Flows : {total_flows:,}")
    print(f"Normal Flows (BENIGN) : {total_flows - detected_attacks:,} ({(total_flows - detected_attacks)/total_flows*100:.2f}%)")
    print(f"Attacks Detected      : {detected_attacks:,} ({detected_attacks/total_flows*100:.2f}%)")
    print("==========================================================")
    
    # 4. حفظ التنبيهات
    df_processed['IDS_Prediction'] = np.where(hybrid_preds == 1, "ATTACK", "BENIGN")
    if detected_attacks > 0:
        alerts = df_processed[df_processed['IDS_Prediction'] == "ATTACK"]
        alerts.to_csv("ids_alerts_log.csv", index=False)
        print("🚨 Attack alerts saved to 'ids_alerts_log.csv'.")

# =====================================================================================
# 4. دالة فحص الروابط (Phishing Detection Function)
# =====================================================================================
def extract_url_features(url):
    url_str = str(url).strip()
    parsed = urlparse(url_str)
    domain = parsed.netloc if parsed.netloc else parsed.path.split('/')[0]

    features = {
        'url_length': len(url_str),
        'domain_length': len(domain),
        'count_dots': url_str.count('.'),
        'count_hyphens': url_str.count('-'),
        'count_underline': url_str.count('_'),
        'count_slash': url_str.count('/'),
        'count_question': url_str.count('?'),
        'count_equal': url_str.count('='),
        'count_at': url_str.count('@'),
        'is_ip': 1 if re.search(r"(([01]?\d\d?|2[0-4]\d|25[0-5])\.){3}([01]?\d\d?|2[0-4]\d|25[0-5])", domain) else 0,
        'is_https': 1 if parsed.scheme.lower() == 'https' else 0,
        'is_shortened': 1 if re.search(r"(bit\.ly|goo\.gl|tinyurl|t\.co)", url_str.lower()) else 0,
        'suspicious_keyword_count': sum(1 for kw in ['login', 'verify', 'update', 'account', 'bank', 'paypal'] if kw in url_str.lower()),
        'subdomain_count': len(domain.split('.')) - 2 if len(domain.split('.')) > 2 else 0,
        'has_double_slash_redirect': 1 if url_str.rfind('//') > 7 else 0
    }
    return features

def check_url_phishing(url):
    feats = pd.DataFrame([extract_url_features(url)])
    feats = feats[phishing_feature_names]
    pred = phishing_model.predict(feats)[0]
    prob = phishing_model.predict_proba(feats)[0]
    
    if pred == 1:
        print(f"🚨 PHISHING ALERT: '{url}' (Confidence: {prob[1]*100:.1f}%)")
    else:
        print(f"✅ BENIGN URL: '{url}' (Confidence: {prob[0]*100:.1f}%)")

# =====================================================================================
# 5. التنفيذ المباشر
# =====================================================================================
if __name__ == "__main__":
    TEST_FILE = "test_set.csv"
    if os.path.exists(TEST_FILE):
        run_ids_detection(TEST_FILE)
        
    print("\n--------------------------------------------------")
    print("Testing Phishing Detector on Sample URLs...")
    print("--------------------------------------------------")
    check_url_phishing("https://www.google.com")
    check_url_phishing("http://192.168.0.1/bank-login")