# ML-Titanic-Project
<div align="center">

# 🚢 Titanic Survival Predictor
### *Uncovering the secrets of the most famous shipwreck*

[![Python]([https://img.shields.io]<img width="100" height="100" alt="image" round= "15" src="https://github.com/user-attachments/assets/401b0149-e7d7-4a66-acfe-517e66bdc8cf" />
))](https://www.python.org)
[![Scikit-Learn](https://img.shields.io)](https://scikit-learn.org)
[![Kaggle](https://img.shields.io)](https://www.kaggle.com)
[![Status](https://img.shields.io)](https://github.com)

---

<p align="left">
  <b>هل تساءلت يوماً من كان بإمكانه النجاة؟</b> <br>
  هذا المشروع ليس مجرد كود، بل هو رحلة تحليلية في بيانات ركاب تايتانيك لاستخدام الذكاء الاصطناعي في التنبؤ بمصير الركاب بناءً على الطبقة، العمر، والجنس.
</p>

[📊 عرض النتائج](#-النتائج) • [🛠️ الأدوات](#️-الأدوات-المستخدمة) • [🚀 التشغيل](#-كيفية-التشغيل)

</div>

## 🔍 نظرة سريعة
*   **الهدف:** بناء نموذج تصنيف (Classification) بدقة عالية.
*   **البيانات:** مجموعة بيانات [Kaggle Titanic](https://www.kaggle.com/data).
*   **الخوارزمية:** تم استخدام **Random Forest Classifier** كونه الأفضل في التعامل مع الميزات المتنوعة.

## 🛠️ الأدوات المستخدمة
| الأداة | الوظيفة |
| :--- | :--- |
| **Pandas** | تنظيف ومعالجة البيانات |
| **Seaborn** | تحليل البيانات بصرياً (EDA) |
| **Scikit-Learn** | بناء وتدريب النموذج البرمجي |

## 📊 النتائج (Insights)
بناءً على التحليل، كانت العوامل الأكثر تأثيراً في النجاة هي:
1.  **الجنس:** النساء كان لهن الأولوية القصوى في النجاة.
2.  **الطبقة:** ركاب الطبقة الأولى (First Class) لديهم معدلات نجاة أعلى بكثير.
3.  **العمر:** الأطفال والشباب كانت فرصهم أفضل.

## 🚀 كيفية التشغيل
```bash
# 1. استنساخ المشروع
git clone https://github.comyour-username/titanic-project.git

# 2. تثبيت المكتبات
pip install -r requirements.txt

# 3. تشغيل الكود
python main.py

### 2️⃣ كود المشروع (Clean & Professional Code)
هذا الكود مكتوب بأسلوب "Production-Ready" ومنظم بشكل يبهر أي مبرمج يراه:

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. تحميل البيانات
def load_data():
    # تأكد من تحميل الملفات من كاجل أو وضعها في نفس المجلد
    train = pd.read_csv('train.csv')
    return train

# 2. تنظيف البيانات (Feature Engineering)
def preprocess_data(df):
    # ملء القيم المفقودة
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    
    # تحويل البيانات النصية لأرقام
    df['Sex'] = df['Sex'].map({'female': 1, 'male': 0})
    
    # اختيار الميزات المهمة فقط
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
    return df[features], df['Survived']

# 3. التدريب والتقييم
if __name__ == "__main__":
    print("🚢 بدأت عملية تحليل بيانات تايتانيك...")
    
    data = load_data()
    X, y = preprocess_data(data)
    
    # تقسيم البيانات
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # بناء النموذج (Random Forest)
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    
    # النتائج
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    
    print(f"✅ تم التدريب بنجاح! الدقة المحققة: {acc:.2%}")
    print("\n--- تقرير التصنيف ---")
    print(classification_report(y_test, predictions))
