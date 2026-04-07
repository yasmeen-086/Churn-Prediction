
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, classification_report,
    confusion_matrix, roc_curve, ConfusionMatrixDisplay
)
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')


def load_data(filepath: str = None) -> pd.DataFrame:
    """
    Load the Telco Customer Churn dataset.
    If no filepath provided, generates a realistic synthetic dataset
    that mirrors the original Kaggle dataset structure.
    """
    if filepath:
        df = pd.read_csv(filepath)
        print(f"✅ Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
        return df

    print("📦 Generating synthetic Telco Churn dataset (mirrors Kaggle structure)...")
    np.random.seed(42)
    n = 7043

    # Demographics
    gender = np.random.choice(['Male', 'Female'], n)
    senior_citizen = np.random.choice([0, 1], n, p=[0.84, 0.16])
    partner = np.random.choice(['Yes', 'No'], n)
    dependents = np.random.choice(['Yes', 'No'], n, p=[0.30, 0.70])

    # Account info
    tenure = np.random.randint(0, 72, n)
    contract = np.random.choice(
        ['Month-to-month', 'One year', 'Two year'], n, p=[0.55, 0.24, 0.21]
    )
    paperless_billing = np.random.choice(['Yes', 'No'], n, p=[0.59, 0.41])
    payment_method = np.random.choice(
        ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'],
        n, p=[0.34, 0.23, 0.22, 0.21]
    )
    monthly_charges = np.round(np.random.uniform(18, 118, n), 2)
    total_charges = np.round(tenure * monthly_charges * np.random.uniform(0.95, 1.05, n), 2)
    total_charges = np.where(tenure == 0, 0, total_charges)

    # Services
    phone_service = np.random.choice(['Yes', 'No'], n, p=[0.90, 0.10])
    multiple_lines = np.where(
        phone_service == 'No', 'No phone service',
        np.random.choice(['Yes', 'No'], n)
    )
    internet_service = np.random.choice(
        ['DSL', 'Fiber optic', 'No'], n, p=[0.34, 0.44, 0.22]
    )

    def internet_addon(internet_service, yes_prob=0.3):
        return np.where(
            internet_service == 'No', 'No internet service',
            np.random.choice(['Yes', 'No'], n, p=[yes_prob, 1 - yes_prob])
        )

    online_security = internet_addon(internet_service, 0.28)
    online_backup = internet_addon(internet_service, 0.34)
    device_protection = internet_addon(internet_service, 0.34)
    tech_support = internet_addon(internet_service, 0.29)
    streaming_tv = internet_addon(internet_service, 0.38)
    streaming_movies = internet_addon(internet_service, 0.39)

    # Churn label — engineered to reflect real-world patterns
    churn_prob = (
        0.05
        + 0.30 * (contract == 'Month-to-month')
        + 0.10 * (internet_service == 'Fiber optic')
        - 0.15 * (tenure > 24)
        + 0.08 * (monthly_charges > 80)
        - 0.06 * (online_security == 'Yes')
        - 0.05 * (tech_support == 'Yes')
        + 0.05 * (senior_citizen == 1)
        + np.random.normal(0, 0.05, n)
    )
    churn_prob = np.clip(churn_prob, 0, 1)
    churn = np.where(np.random.random(n) < churn_prob, 'Yes', 'No')

    df = pd.DataFrame({
        'customerID': [f'CUST-{i:05d}' for i in range(n)],
        'gender': gender,
        'SeniorCitizen': senior_citizen,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'Churn': churn,
    })

    print(f"✅ Dataset generated: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


# 2. DATA PREPROCESSING


def preprocess(df: pd.DataFrame):
    """Clean, encode, and engineer features."""
    df = df.copy()
    df.drop(columns=['customerID'], inplace=True, errors='ignore')

    # Fix TotalCharges
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(0, inplace=True)

    # Feature Engineering 
    df['tenure_ratio'] = df['tenure'] / (df['tenure'].max() + 1)

    contract_map = {'Month-to-month': 1, 'One year': 12, 'Two year': 24}
    df['contract_duration'] = df['Contract'].map(contract_map)

    df['avg_monthly_spend'] = np.where(
        df['tenure'] > 0,
        df['TotalCharges'] / df['tenure'],
        df['MonthlyCharges']
    )

    service_cols = [
        'PhoneService', 'MultipleLines', 'OnlineSecurity',
        'OnlineBackup', 'DeviceProtection', 'TechSupport',
        'StreamingTV', 'StreamingMovies'
    ]
    df['num_services'] = sum(
        (df[col] == 'Yes').astype(int) for col in service_cols
    )

    df['is_senior_no_partner'] = (
        (df['SeniorCitizen'] == 1) & (df['Partner'] == 'No')
    ).astype(int)

    
    df['Churn'] = (df['Churn'] == 'Yes').astype(int)

    binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    for col in binary_cols:
        df[col] = LabelEncoder().fit_transform(df[col])

    df = pd.get_dummies(df, columns=[
        'MultipleLines', 'InternetService', 'OnlineSecurity',
        'OnlineBackup', 'DeviceProtection', 'TechSupport',
        'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod'
    ], drop_first=True)

    print(f"✅ After preprocessing: {df.shape[1]} features")
    return df


# 3. EDA PLOTS

def plot_eda(df_raw: pd.DataFrame):
    """Generate Exploratory Data Analysis charts."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('Exploratory Data Analysis — Customer Churn', fontsize=16, fontweight='bold', y=1.01)

    palette = {'No': '#4C9BE8', 'Yes': '#E85D5D'}

    # 1. Churn distribution
    churn_counts = df_raw['Churn'].value_counts()
    axes[0, 0].bar(churn_counts.index, churn_counts.values,
                   color=[palette['No'], palette['Yes']], edgecolor='white', linewidth=1.5)
    axes[0, 0].set_title('Churn Distribution')
    axes[0, 0].set_xlabel('Churn')
    axes[0, 0].set_ylabel('Count')
    for i, v in enumerate(churn_counts.values):
        axes[0, 0].text(i, v + 50, f'{v:,}', ha='center', fontweight='bold')

    # 2. Tenure by churn
    for label, color in palette.items():
        subset = df_raw[df_raw['Churn'] == label]['tenure']
        axes[0, 1].hist(subset, bins=30, alpha=0.7, label=label, color=color)
    axes[0, 1].set_title('Tenure Distribution by Churn')
    axes[0, 1].set_xlabel('Tenure (months)')
    axes[0, 1].legend()

    # 3. Monthly charges by churn
    for label, color in palette.items():
        subset = df_raw[df_raw['Churn'] == label]['MonthlyCharges']
        axes[0, 2].hist(subset, bins=30, alpha=0.7, label=label, color=color)
    axes[0, 2].set_title('Monthly Charges by Churn')
    axes[0, 2].set_xlabel('Monthly Charges ($)')
    axes[0, 2].legend()

    # 4. Churn by Contract
    contract_churn = df_raw.groupby('Contract')['Churn'].value_counts(normalize=True).unstack()
    contract_churn['Yes'].sort_values(ascending=True).plot(
        kind='barh', ax=axes[1, 0], color='#E85D5D'
    )
    axes[1, 0].set_title('Churn Rate by Contract Type')
    axes[1, 0].set_xlabel('Churn Rate')
    axes[1, 0].set_xlim(0, 0.6)

    # 5. Churn by Internet Service
    inet_churn = df_raw.groupby('InternetService')['Churn'].value_counts(normalize=True).unstack()
    inet_churn.plot(kind='bar', ax=axes[1, 1],
                    color=[palette['No'], palette['Yes']], edgecolor='white')
    axes[1, 1].set_title('Churn Rate by Internet Service')
    axes[1, 1].set_xticklabels(axes[1, 1].get_xticklabels(), rotation=15)
    axes[1, 1].legend(['No Churn', 'Churn'])

    # 6. ✅ FIXED Correlation heatmap
    temp = df_raw.copy()

    temp['TotalCharges'] = pd.to_numeric(temp['TotalCharges'], errors='coerce')

    numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen']
    for col in numeric_cols:
        temp[col] = pd.to_numeric(temp[col], errors='coerce')

    # Encode target
    temp['Churn_num'] = (temp['Churn'] == 'Yes').astype(int)

    # Compute correlation
    corr = temp[numeric_cols + ['Churn_num']].corr()

    sns.heatmap(corr, ax=axes[1, 2], annot=True, fmt='.2f',
                cmap='RdBu_r', center=0, linewidths=0.5)
    axes[1, 2].set_title('Correlation Heatmap')

    plt.tight_layout()
    plt.savefig('eda_plots.png', dpi=150, bbox_inches='tight')
    print("📊 EDA saved → eda_plots.png")
    plt.show()



# 4. MODEL TRAINING

def train_models(X_train, X_test, y_train, y_test):
    """Train Logistic Regression and Random Forest with scaling pipeline."""

    results = {}

    print("\n🔵 Training Logistic Regression...")
    lr_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'))
    ])
    lr_pipe.fit(X_train, y_train)

    lr_pred = lr_pipe.predict(X_test)
    lr_proba = lr_pipe.predict_proba(X_test)[:, 1]
    lr_cv = cross_val_score(lr_pipe, X_train, y_train, cv=5, scoring='roc_auc')

    results['Logistic Regression'] = {
        'model': lr_pipe,
        'predictions': lr_pred,
        'probabilities': lr_proba,
        'accuracy': accuracy_score(y_test, lr_pred),
        'roc_auc': roc_auc_score(y_test, lr_proba),
        'cv_auc_mean': lr_cv.mean(),
        'cv_auc_std': lr_cv.std(),
    }

    #  Random Forest
    print("🌲 Training Random Forest...")
    rf_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ))
    ])
    rf_pipe.fit(X_train, y_train)

    rf_pred = rf_pipe.predict(X_test)
    rf_proba = rf_pipe.predict_proba(X_test)[:, 1]
    rf_cv = cross_val_score(rf_pipe, X_train, y_train, cv=5, scoring='roc_auc')

    results['Random Forest'] = {
        'model': rf_pipe,
        'predictions': rf_pred,
        'probabilities': rf_proba,
        'accuracy': accuracy_score(y_test, rf_pred),
        'roc_auc': roc_auc_score(y_test, rf_proba),
        'cv_auc_mean': rf_cv.mean(),
        'cv_auc_std': rf_cv.std(),
    }

   
    print("\n" + "="*55)
    print(f"{'Model':<22} {'Accuracy':>10} {'ROC-AUC':>10} {'CV AUC':>12}")
    print("="*55)
    for name, r in results.items():
        print(f"{name:<22} {r['accuracy']:>10.4f} {r['roc_auc']:>10.4f} "
              f"{r['cv_auc_mean']:.4f}±{r['cv_auc_std']:.4f}")
    print("="*55)

    return results



def plot_results(results: dict, X_test, y_test, feature_names):
    """Plot confusion matrices, ROC curves, and feature importances."""

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle('Model Evaluation — Customer Churn Prediction', fontsize=16, fontweight='bold')

    colors = {'Logistic Regression': '#4C9BE8', 'Random Forest': '#2ECC71'}

    for idx, (name, r) in enumerate(results.items()):
        cm = confusion_matrix(y_test, r['predictions'])
        disp = ConfusionMatrixDisplay(cm, display_labels=['Stay', 'Churn'])
        disp.plot(ax=axes[0, idx], colorbar=False, cmap='Blues')
        axes[0, idx].set_title(f'{name}\nAccuracy: {r["accuracy"]:.4f}')

    ax_roc = axes[0, 2]
    for name, r in results.items():
        fpr, tpr, _ = roc_curve(y_test, r['probabilities'])
        ax_roc.plot(fpr, tpr, label=f'{name} (AUC={r["roc_auc"]:.4f})',
                    color=colors[name], linewidth=2.5)
    ax_roc.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random Baseline')
    ax_roc.set_title('ROC Curves')
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.legend(loc='lower right')
    ax_roc.grid(True, alpha=0.3)

    for idx, (name, r) in enumerate(results.items()):
        report = classification_report(y_test, r['predictions'],
                                       target_names=['Stay', 'Churn'], output_dict=True)
        metrics = ['precision', 'recall', 'f1-score']
        classes = ['Stay', 'Churn']
        data = [[report[c][m] for m in metrics] for c in classes]
        im = axes[1, idx].imshow(data, cmap='YlGn', vmin=0, vmax=1, aspect='auto')
        axes[1, idx].set_xticks(range(len(metrics)))
        axes[1, idx].set_xticklabels(metrics)
        axes[1, idx].set_yticks(range(len(classes)))
        axes[1, idx].set_yticklabels(classes)
        axes[1, idx].set_title(f'{name} — Classification Report')
        for i in range(len(classes)):
            for j in range(len(metrics)):
                axes[1, idx].text(j, i, f'{data[i][j]:.3f}', ha='center',
                                  va='center', fontweight='bold', fontsize=12)
        plt.colorbar(im, ax=axes[1, idx])

    rf_model = results['Random Forest']['model'].named_steps['clf']
    importances = pd.Series(rf_model.feature_importances_, index=feature_names)
    top15 = importances.nlargest(15).sort_values()
    ax_fi = axes[1, 2]
    bars = ax_fi.barh(range(len(top15)), top15.values, color='#2ECC71', edgecolor='white')
    ax_fi.set_yticks(range(len(top15)))
    ax_fi.set_yticklabels(top15.index, fontsize=9)
    ax_fi.set_title('Top 15 Feature Importances (Random Forest)')
    ax_fi.set_xlabel('Importance')
    ax_fi.grid(True, axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig('model_results.png', dpi=150, bbox_inches='tight')
    print("📊 Results saved → model_results.png")
    plt.show()




def main():
    print("=" * 55)
    print("   CUSTOMER CHURN PREDICTION PIPELINE")
    print("   Python | Pandas | Scikit-learn")
    print("=" * 55)

    # Step 1: Load data
    df_raw = load_data('/Users/jass/Downloads/WA_Fn-UseC_-Telco-Customer-Churn.csv')

    # Step 2: EDA
    plot_eda(df_raw)

    # Step 3: Preprocess
    df = preprocess(df_raw)

    X = df.drop(columns=['Churn'])
    y = df['Churn']
    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n📂 Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"📊 Churn rate (train): {y_train.mean():.2%} | (test): {y_test.mean():.2%}")

    # Step 4: Train models
    results = train_models(X_train, X_test, y_train, y_test)

    # Step 5: Plot results
    plot_results(results, X_test, y_test, feature_names)

    # Step 6: Summary
    best_name = max(results, key=lambda k: results[k]['roc_auc'])
    best = results[best_name]
    print(f"\n Best Model: {best_name}")
    print(f"    Accuracy : {best['accuracy']:.4f}")
    print(f"    ROC-AUC  : {best['roc_auc']:.4f}")
    print(f"    CV AUC   : {best['cv_auc_mean']:.4f} ± {best['cv_auc_std']:.4f}")

    return results


if __name__ == '__main__':
    results = main()
