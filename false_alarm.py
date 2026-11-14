import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# ==========================================
# 1. بناء طبقة فوضوية (Chaotic Activation/Weight Layer)
# ==========================================
class ChaoticNeuronLayer(tf.keras.layers.Layer):
    """
    طبقة مخصصة تعتمد على Logistic Map الفوضوي: x_{n+1} = r * x_n * (1 - x_n)
    تُستخدم لإضافة ديناميكية فوضوية تسهم في حسم القرارات المشكوك فيها تقليلاً للإنذارات الكاذبة.
    """
    def __init__(self, units, r=3.99, **kwargs):
        super(ChaoticNeuronLayer, self).__init__(**kwargs)
        self.units = units
        self.r = r  # المعامل الفوضوي (r close to 4 produces chaos)

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="glorot_uniform",
            trainable=True,
            name="chaotic_weights"
        )
        self.b = self.add_weight(
            shape=(self.units,),
            initializer="zeros",
            trainable=True,
            name="chaotic_bias"
        )

    def call(self, inputs):
        # حساب الدخل الخطي
        linear_out = tf.matmul(inputs, self.w) + self.b
        
        # تطبيق خريطة اللوجستيك الفوضوية على قيم الدخل المعيرة بين (0, 1)
        normalized_inputs = tf.sigmoid(linear_out)
        chaotic_state = self.r * normalized_inputs * (1.0 - normalized_inputs)
        
        # الدمج بين التنشيط الطبيعي والاستجابة الفوضوية
        return tf.sigmoid(linear_out + chaotic_state)

# ==========================================
# 2. إنشاء بيانات تجريبية (أو تحميل بياناتك النظيفة)
# ==========================================
print("جاري إعداد وتجهيز البيانات...")
# توليد عينة بيانات ممثلة لتجربة المعمارية
np.random.seed(42)
n_samples = 10000
n_features = 20

X_dummy = np.random.randn(n_samples, n_features)
# 80% طبيعي (0) و 20% هجمات (1)
y_dummy = np.random.choice([0, 1], size=n_samples, p=[0.8, 0.2])

# تقسيم البيانات
X_train, X_test, y_train, y_test = train_test_split(
    X_dummy, y_dummy, test_size=0.3, random_state=42, stratify=y_dummy
)

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# 3. النموذج الأول: MLP للكشف الأول عالي الحساسية
# ==========================================
print("\n--- 1. تدريب نموذج MLP لكشف الهجمات الأولية ---")
mlp_model = Sequential([
    Dense(64, activation='relu', input_shape=(n_features,)),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

mlp_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
mlp_model.fit(X_train_scaled, y_train, epochs=5, batch_size=256, verbose=1)

# تنبؤات الـ MLP الأولية
mlp_preds_prob = mlp_model.predict(X_test_scaled)
mlp_preds = (mlp_preds_prob >= 0.5).astype(int).flatten()

# ==========================================
# 4. النموذج الثاني: Chaotic Neural Network (لتقليل الإنذارات الكاذبة)
# ==========================================
print("\n--- 2. تدريب النموذج الفوضوي (Chaotic Filter) لتقليل FP ---")

# بناء نموذج الفلتر الفوضوي
chaotic_model = Sequential([
    Dense(32, activation='relu', input_shape=(n_features,)),
    ChaoticNeuronLayer(units=16, r=3.99), # الطبقة الفوضوية
    Dense(1, activation='sigmoid')
])

chaotic_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# تدريب النموذج الفوضوي مركزاً على العينات الحساسة والمهمة
chaotic_model.fit(X_train_scaled, y_train, epochs=5, batch_size=256, verbose=1)

# ==========================================
# 5. منطق الدمج الهجين (Hybrid Pipeline)
# ==========================================
print("\n--- 3. تطبيق منطق الكشف الهجين وسلسلة الفلترة ---")

final_predictions = []

# تتبع الإحصائيات
false_positives_reduced = 0

for i in range(len(X_test_scaled)):
    mlp_decision = mlp_preds[i]
    
    # إذا صنف MLP الحزمة على أنها "طبيعية"، نعتمدها مباشرة
    if mlp_decision == 0:
        final_predictions.append(0)
    else:
        # إذا صنف MLP الحزمة على أنها "هجوم"، نمررها للفلتر الفوضوي للتأكد
        sample = X_test_scaled[i:i+1]
        chaotic_prob = chaotic_model.predict(sample, verbose=0)[0][0]
        
        # شرط الفلتر الفوضوي لتثبيت الهجوم أو إلغائه (تقليل الانذارات الكاذبة)
        if chaotic_prob >= 0.6:  # عتبة تصديق أعلى للحد من التنبيه الخاطئ
            final_predictions.append(1)
        else:
            final_predictions.append(0)
            if y_test[i] == 0:
                false_positives_reduced += 1

final_predictions = np.array(final_predictions)

# ==========================================
# 6. تقييم النتائج وعرض الأداء
# ==========================================
print("\n================ نتائج النظام الهجين (MLP + Chaotic Network) ================")
print(f"عدد الإنذارات الكاذبة التي تم تصحيحها وإلغاؤها بواسطة الفلتر الفوضوي: {false_positives_reduced}")
print(f"الدقة الإجمالية النهائية: {accuracy_score(y_test, final_predictions) * 100:.2f}%\n")

cm = confusion_matrix(y_test, final_predictions)
print("مصفوفة الارتباك (Confusion Matrix):")
print(f"True Negatives (طبيعي صح): {cm[0][0]}")
print(f"False Positives (إنذار كاذب): {cm[0][1]}")
print(f"False Negatives (هجوم مر): {cm[1][0]}")
print(f"True Positives (هجوم مكتشف): {cm[1][1]}\n")

print("تقرير التصنيف التفصيلي:")
print(classification_report(y_test, final_predictions, target_names=["Benign (طبيعي)", "Attack (هجوم)"]))