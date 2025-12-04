# StreamGuard
Quality of Experience (QoE) Anomaly Detection.
StreamGuard: Quality of Experience (QoE) Anomaly Detection

📌 Project Overview

StreamGuard is a telemetry analytics engine designed to detect "Silent Failures" in video streaming sessions. Unlike hard errors (e.g., 404s), silent failures occur when a user experiences a degraded viewing experience—such as high buffering or pixelated bitrate—while the application remains technically functional.

This project simulates 10,000 real-time session logs and utilizes unsupervised machine learning (Isolation Forest) to identify anomalous patterns in Quality of Experience (QoE) metrics.

🚀 Business Value

For streaming platforms, retention is directly correlated with viewing quality. This tool aids in:

Proactive Alerting: Detecting regional ISP issues before customer support tickets spike.

Device Health Monitoring: Identifying specific device models (e.g., "Legacy Smart TVs") struggling with new encoding formats.

Churn Prevention: Flagging sessions with poor QoE for potential customer save campaigns.

🛠️ Tech Stack

Language: Python 3.9+

Data Processing: Pandas, NumPy

Machine Learning: Scikit-Learn (Isolation Forest)

Visualization: Seaborn, Matplotlib

📂 Project Structure

StreamGuard/
├── stream_guard.py        # Main ETL and Anomaly Detection logic
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
└── anomaly_visualization.png # Generated scatterplot of detected clusters


⚙️ How to Run

Clone the repository (or download files):

git clone [https://github.com/yourusername/StreamGuard-Anomaly-Detection.git](https://github.com/yourusername/StreamGuard-Anomaly-Detection.git)
cd StreamGuard-Anomaly-Detection


Install Dependencies:

pip install -r requirements.txt


Execute the Engine:

python stream_guard.py


This will generate a simulated dataset, train the model, and output the detection results.

View Results:

Check the console for the anomaly count.

Open anomaly_visualization.png to see the decision boundaries.

📊 Methodology

Data Simulation: Generates synthetic telemetry including Bitrate, BufferingRatio, Region, and DeviceType. 5% of data is injected with fault patterns (e.g., High Buffering + Low Bitrate).

Feature Engineering: Selects high-impact features (buffering_ratio, bitrate_kbps) that directly impact user perception.

Modeling: Uses Isolation Forest, an ensemble tree-based algorithm effective for high-dimensional anomaly detection where "normal" data is abundant and "anomalies" are rare.

Thresholding: Anomalies are flagged with a score of -1 based on a contamination estimate of 5%.

🔮 Future Roadmap

Real-time Ingestion: Integrate Apache Kafka to process session logs in real-time.

Scalability: Migrate the IsolationForest logic to Spark MLlib for distributed processing of petabyte-scale logs.

Root Cause Analysis: Add a supervised layer (e.g., XGBoost) to classify types of anomalies (e.g., "CDN Failure" vs. "Last Mile Issue").

Author: Abhayraj Singh
