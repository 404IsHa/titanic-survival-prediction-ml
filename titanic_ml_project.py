import os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, f1_score
from sklearn.pipeline import Pipeline
import joblib
warnings.filterwarnings('ignore')

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
os.makedirs('outputs', exist_ok=True)
os.makedirs('models', exist_ok=True)
plt.rcParams.update({'figure.dpi': 120, 'savefig.bbox': 'tight',
                     'axes.spines.top': False, 'axes.spines.right': False})
sns.set_theme(style='whitegrid')

print("=" * 55)
print("  Titanic Survival Prediction - ML Mini Project")
print("=" * 55)

print("\n[1/7] Generating data...")
n = 800
pclass  = np.random.choice([1, 2, 3], n, p=[0.24, 0.21, 0.55])
sex_raw = np.random.choice(['male', 'female'], n, p=[0.65, 0.35])
age     = np.random.normal(30, 14, n).clip(1, 80)
age     = pd.array(age, dtype=pd.Float64Dtype())
age[np.random.rand(n) < 0.05] = pd.NA
sibsp   = np.random.choice([0, 1, 2, 3], n, p=[0.68, 0.20, 0.08, 0.04])
parch   = np.random.choice([0, 1, 2, 3], n, p=[0.76, 0.13, 0.08, 0.03])
fare    = np.where(pclass == 1, np.random.lognormal(4.5, 0.7, n),
          np.where(pclass == 2, np.random.lognormal(3.0, 0.5, n),
                                np.random.lognormal(2.0, 0.6, n))).round(2)
emb_raw = np.random.choice(['S', 'C', 'Q'], n, p=[0.72, 0.19, 0.09]).tolist()
for i in np.where(np.random.rand(n) < 0.02)[0]:
    emb_raw[i] = None
survive_prob = np.full(n, 0.38)
survive_prob[sex_raw == 'female'] += 0.35
survive_prob[pclass == 1]         += 0.18
survive_prob[pclass == 3]         -= 0.15
survive_prob = survive_prob.clip(0.05, 0.95)
survived = (np.random.rand(n) < survive_prob).astype(int)
df = pd.DataFrame({'survived': survived, 'pclass': pclass, 'sex': sex_raw,
                   'age': age, 'sibsp': sibsp, 'parch': parch,
                   'fare': fare, 'embarked': emb_raw})
print(f"  Dataset shape  : {df.shape}")
print(f"  Survived count : {df['survived'].value_counts().to_dict()}")

print("\n[2/7] Running EDA...")
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.flatten()
counts = df['survived'].value_counts().sort_index()
axes[0].bar(['Died', 'Survived'], counts.values,
            color=['#e74c3c','#2ecc71'], edgecolor='white', width=0.5)
axes[0].set_title('Overall survival')
for i, v in enumerate(counts.values):
    axes[0].text(i, v+5, str(v), ha='center', fontweight='bold')
sr_sex = df.groupby('sex')['survived'].mean() * 100
axes[1].bar(sr_sex.index, sr_sex.values,
            color=['#3498db','#e74c3c'], edgecolor='white', width=0.4)
axes[1].set_title('Survival rate by sex (%)')
axes[1].set_ylim(0, 100)
for i, v in enumerate(sr_sex.values):
    axes[1].text(i, v+1, f'{v:.1f}%', ha='center', fontweight='bold')
sr_cls = df.groupby('pclass')['survived'].mean() * 100
axes[2].bar([f'Class {c}' for c in sr_cls.index], sr_cls.values,
            color=['#3498db','#9b59b6','#95a5a6'], edgecolor='white', width=0.5)
axes[2].set_title('Survival rate by class (%)')
axes[2].set_ylim(0, 100)
for i, v in enumerate(sr_cls.values):
    axes[2].text(i, v+1, f'{v:.1f}%', ha='center', fontweight='bold')
age_num = pd.to_numeric(df['age'], errors='coerce')
for label, color in [(0,'#e74c3c'),(1,'#2ecc71')]:
    axes[3].hist(age_num[df['survived']==label].dropna(), bins=25,
                 alpha=0.6, color=color, edgecolor='white',
                 label='Survived' if label==1 else 'Died')
axes[3].set_title('Age distribution')
axes[3].legend()
for label, color in [(0,'#e74c3c'),(1,'#2ecc71')]:
    axes[4].hist(np.log1p(df[df['survived']==label]['fare']), bins=25,
                 alpha=0.6, color=color, edgecolor='white',
                 label='Survived' if label==1 else 'Died')
axes[4].set_title('Fare distribution (log)')
axes[4].legend()
num_df = df[['survived','pclass','sibsp','parch','fare']].copy()
sns.heatmap(num_df.corr(), annot=True, fmt='.2f', cmap='RdYlGn',
            center=0, ax=axes[5], linewidths=0.3)
axes[5].set_title('Correlation matrix')
plt.suptitle('Titanic - Exploratory Data Analysis', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/eda.png')
plt.close()
print("  EDA plot saved -> outputs/eda.png")

print("\n[3/7] Preprocessing...")
df['age']      = pd.to_numeric(df['age'], errors='coerce')
df['age']      = df['age'].fillna(df['age'].median())
df['fare']     = df['fare'].fillna(df['fare'].median())
df['embarked'] = df['embarked'].fillna('S')
df['family_size'] = df['sibsp'] + df['parch']
df['is_alone']    = (df['family_size'] == 0).astype(int)
df['fare_log']    = np.log1p(df['fare'])
df['sex_enc']     = LabelEncoder().fit_transform(df['sex'])
df['emb_enc']     = LabelEncoder().fit_transform(df['embarked'])
df['age_bin']     = pd.cut(df['age'], bins=[0,12,18,35,60,100],
                            labels=[0,1,2,3,4]).astype(int)
FEATURES = ['pclass','sex_enc','age','sibsp','parch',
            'fare_log','emb_enc','family_size','is_alone','age_bin']
X = df[FEATURES]
y = df['survived']
print(f"  Features used : {FEATURES}")
print(f"  Final shape   : {X.shape}")

print("\n[4/7] Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
print(f"  Train : {len(X_train)}  |  Test : {len(X_test)}")

print("\n[5/7] Training models...")
models = {
    'Logistic Regression': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
    ]),
    'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=7,
                                            random_state=RANDOM_STATE, n_jobs=-1),
}
trained = {}
for name, model in models.items():
    cv = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    model.fit(X_train, y_train)
    trained[name] = model
    print(f"  {name:25s} CV={cv.mean():.4f} +/-{cv.std():.4f}")

print("\n[6/7] Evaluating models...")
results = {}
for name, model in trained.items():
    preds = model.predict(X_test)
    acc   = accuracy_score(y_test, preds)
    f1    = f1_score(y_test, preds, average='weighted')
    results[name] = {'accuracy': acc, 'f1': f1, 'preds': preds}
    print(f"\n  {name}")
    print(f"  Accuracy={acc:.4f}  F1={f1:.4f}")
    print(classification_report(y_test, preds,
                                 target_names=['Died','Survived'], digits=3))
n_m = len(trained)
fig, axes = plt.subplots(1, n_m+1, figsize=(5*(n_m+1), 5))
palette = ['#3498db','#9b59b6','#2ecc71']
for idx, (name, res) in enumerate(results.items()):
    cm = confusion_matrix(y_test, res['preds'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Died','Survived'],
                yticklabels=['Died','Survived'],
                ax=axes[idx], linewidths=0.5)
    axes[idx].set_title(f'{name}\nAcc={res["accuracy"]:.3f}')
    axes[idx].set_xlabel('Predicted')
    axes[idx].set_ylabel('True')
ax_roc = axes[-1]
for (name, res), color in zip(results.items(), palette):
    model = trained[name]
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(X_test)[:,1]
        fpr, tpr, _ = roc_curve(y_test, probs)
        ax_roc.plot(fpr, tpr, color=color, lw=2,
                    label=f'{name} (AUC={auc(fpr,tpr):.3f})')
ax_roc.plot([0,1],[0,1],'k--',lw=1,alpha=0.5)
ax_roc.set_title('ROC Curves')
ax_roc.set_xlabel('False Positive Rate')
ax_roc.set_ylabel('True Positive Rate')
ax_roc.legend(fontsize=8)
plt.tight_layout()
plt.savefig('outputs/evaluation.png')
plt.close()
print("  Evaluation plot saved -> outputs/evaluation.png")
names_r = list(results.keys())
accs  = [results[n]['accuracy'] for n in names_r]
f1s   = [results[n]['f1']       for n in names_r]
x     = np.arange(len(names_r))
w     = 0.35
fig, ax = plt.subplots(figsize=(9, 5))
b1 = ax.bar(x-w/2, accs, w, label='Accuracy', color='#3498db', edgecolor='white')
b2 = ax.bar(x+w/2, f1s,  w, label='F1 Score', color='#9b59b6', edgecolor='white')
for bar in list(b1)+list(b2):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
            f'{bar.get_height():.3f}', ha='center', fontsize=9, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(names_r)
ax.set_ylim(0, 1.12)
ax.set_ylabel('Score')
ax.set_title('Model Comparison', fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig('outputs/model_comparison.png')
plt.close()
print("  Comparison plot saved -> outputs/model_comparison.png")

print("\n[7/7] Feature importance & saving model...")
rf = trained['Random Forest']
importances = rf.feature_importances_
indices     = np.argsort(importances)[::-1]
fig, ax = plt.subplots(figsize=(9, 5))
ax.barh([FEATURES[i] for i in indices][::-1],
         importances[indices][::-1],
         color='#3498db', edgecolor='white')
ax.set_xlabel('Importance')
ax.set_title('Feature Importance - Random Forest', fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/feature_importance.png')
plt.close()
print("  Feature importance saved -> outputs/feature_importance.png")
print("\n  Top 3 most important features:")
for i in range(3):
    print(f"    {i+1}. {FEATURES[indices[i]]:15s} {importances[indices[i]]:.4f}")
joblib.dump(rf, 'models/random_forest.joblib')
print("  Model saved -> models/random_forest.joblib")

print("\n  Predictions for new passengers:")
new = pd.DataFrame([
    {'pclass':1,'sex_enc':0,'age':20,'sibsp':1,'parch':0,
     'fare_log':np.log1p(150),'emb_enc':0,'family_size':1,'is_alone':0,'age_bin':2},
    {'pclass':3,'sex_enc':1,'age':20,'sibsp':0,'parch':0,
     'fare_log':np.log1p(8),'emb_enc':0,'family_size':0,'is_alone':1,'age_bin':2},
    {'pclass':2,'sex_enc':0,'age':8,'sibsp':1,'parch':2,
     'fare_log':np.log1p(30),'emb_enc':0,'family_size':3,'is_alone':0,'age_bin':0},
])
preds = rf.predict(new)
probs = rf.predict_proba(new)
names_p = ['Rose (1st class, female, 20)',
           'Jack (3rd class, male,   20)',
           'Child (2nd class, female, 8)']
for name, pred, prob in zip(names_p, preds, probs):
    label = 'SURVIVED' if pred==1 else 'DIED'
    print(f"  {'[+]' if pred==1 else '[-]'} {name:35s} {label:10s} ({prob[1]:.1%} chance)")

print("\n" + "="*55)
print("  DONE! Check the outputs/ folder for your plots")
print("="*55)
