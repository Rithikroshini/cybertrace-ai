## 🚀 Live Demo
CyberTrace AI is deployed on Render.  
👉 [Click here to view the app](https://cybertrace-ai-6i5r.onrender.com)

# CyberTrace AI – Predictive Cybercrime Cash-Out & Location Analytics

CyberTrace AI is a machine learning-powered decision support platform built for law enforcement and cybercrime investigators. When financial fraud or phishing occurs, cyber fraudsters frequently utilize mule bank accounts to cash out funds via nearby physical ATMs within minutes. CyberTrace AI predicts the most likely cash-out ATM locations using spatial and situational ML classification.

---

## 🌟 Key Features
- **Machine Learning Classification Engine:** Uses a `RandomForestClassifier` trained on multi-dimensional incident parameters (`victim coordinates`, `fraud amount`, `scam category`, `incident hour`).
- **Feature-Based Synthetic Dataset Modeling:** Training dataset target labels are generated using spatial distance decay (distance in km), transit hub preferences, and night-time operation boosts.
- **Interactive Geospatial Dashboard:** Built with Leaflet.js and CARTO Dark basemaps allowing single-click coordinate pinning.
- **Exportable Forensic Incident Reports:** Generates client-side PDF forensic reports using `jsPDF`.
- **Transparent Architecture:** Clearly distinguishes between active ML inferences and simulated telemetry feeds for hackathon evaluation clarity.

---

## 🛠️ Tech Stack
- **Backend:** Python, Flask, Pandas, NumPy, Scikit-learn  
- **Frontend:** HTML5, Tailwind CSS, JavaScript (ES6)  
- **Mapping & Charts:** Leaflet.js, Chart.js, Lucide Icons, jsPDF  

---

## 🚀 How to Run Locally

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/cybertrace-ai.git
   cd cybertrace-ai
