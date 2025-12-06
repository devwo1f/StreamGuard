import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. DATA SIMULATION (The "Generator")
# ==========================================
def generate_telemetry_data(n_samples=10000, anomaly_fraction=0.05):
    """
    Simulates streaming logs.
    - Normal behavior: High bitrate, near-zero buffering.
    - Anomalous behavior: Low bitrate, frequent buffering.
    """
    n_outliers = int(n_samples * anomaly_fraction)
    n_inliers = n_samples - n_outliers

    # --- Generate Normal Traffic (Inliers) ---
    # Bitrate: Normal distribution centered at 5000kbps (1080p quality)
    normal_bitrate = np.random.normal(loc=5000, scale=1000, size=n_inliers)
    
    # Buffering: Exponential distribution (most people have 0, few have little)
    # scale=0.1 means avg buffering is very low
    normal_buffering = np.random.exponential(scale=0.05, size=n_inliers)

    # --- Generate "Silent Failures" (Outliers) ---
    # Scenario: Congested Network (Low Bitrate + High Buffering)
    anomaly_bitrate = np.random.normal(loc=800, scale=200, size=n_outliers) 
    anomaly_buffering = np.random.uniform(low=0.10, high=0.50, size=n_outliers) 

    # Combine data
    bitrate = np.concatenate([normal_bitrate, anomaly_bitrate])
    buffering = np.concatenate([normal_buffering, anomaly_buffering])
    
    # Create DataFrame
    df = pd.DataFrame({
        'session_id': range(n_samples),
        'bitrate_kbps': bitrate,
        'buffering_ratio': buffering
    })

    # Cleanup: Ensure no negative values (statistical artifact protection)
    df['bitrate_kbps'] = df['bitrate_kbps'].clip(lower=0)
    df['buffering_ratio'] = df['buffering_ratio'].clip(lower=0, upper=1)

    # Shuffle dataset so anomalies aren't all at the bottom
    return df.sample(frac=1, random_state=42).reset_index(drop=True)

# ==========================================
# 2. MODEL TRAINING (The "Brain")
# ==========================================
def train_monitor(df):
    """
    Trains an Isolation Forest to detect anomalies.
    """
    # Select features for the model
    features = ['bitrate_kbps', 'buffering_ratio']
    X = df[features]

    # Initialize Isolation Forest
    # contamination=0.05 tells the model we expect ~5% of data to be bad.
    iso_forest = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
    
    # Fit and Predict
    # Returns 1 for Normal, -1 for Anomaly
    df['anomaly_score'] = iso_forest.fit_predict(X)
    
    return df

# ==========================================
# 3. VISUALIZATION (The "Proof")
# ==========================================
def visualize_results(df):
    plt.figure(figsize=(10, 6))
    
    # Scatter plot
    sns.scatterplot(
        data=df, 
        x='bitrate_kbps', 
        y='buffering_ratio', 
        hue='anomaly_score', 
        palette={1: 'blue', -1: 'red'},
        alpha=0.6
    )
    
    plt.title('StreamGuard: QoE Anomaly Detection')
    plt.xlabel('Bitrate (kbps)')
    plt.ylabel('Buffering Ratio (0.0 - 1.0)')
    plt.axvline(x=2000, color='gray', linestyle='--', alpha=0.5, label='Quality Threshold')
    plt.legend(title='Status (1=Normal, -1=Anomaly)')
    plt.show()

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    print("🚀 Starting StreamGuard Simulation...")
    
    # 1. Generate Data
    data = generate_telemetry_data()
    print(f"Generated {len(data)} session logs.")

    # 2. Detect Anomalies
    results = train_monitor(data)
    
    anomalies = results[results['anomaly_score'] == -1]
    print(f"⚠️ Detected {len(anomalies)} silent failures.")
    print("Sample Anomalies:")
    print(anomalies.head())

    # 3. Visualize
    print("📊 Generating Visualization...")
    visualize_results(results)
