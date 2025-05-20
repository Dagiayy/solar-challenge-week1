
# Solar Farm Potential Analysis Dashboard

## Overview
This Streamlit app visualizes solar potential metrics across Benin, Sierra Leone, and Togo to identify high-potential regions for solar installations. It provides interactive features such as filters, visualizations, and actionable insights to help stakeholders make informed decisions.

The app is designed to load actual cleaned CSV files (`benin_clean.csv`, `sierraleone_clean.csv`, `togo_clean.csv`) if available. If the actual data files are missing, the app generates realistic mock data for demonstration purposes.

---

## Table of Contents
1. [Features](#features)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Folder Structure](#folder-structure)
5. [Running the App](#running-the-app)
6. [Development Process](#development-process)
7. [Key Performance Indicators (KPIs)](#key-performance-indicators-kpis)
8. [Deployment](#deployment)
9. [Contributing](#contributing)
10. [License](#license)

---

## Features
- **Interactive Filters**: Select countries, metrics, and advanced filters to customize analysis.
- **Visualizations**:
  - Boxplots: Compare key metrics across countries.
  - Bubble Charts: Explore relationships between multiple variables.
  - Correlation Heatmaps: Analyze relationships between environmental variables.
  - Regional Rankings: Identify top-performing regions.
- **Insights & Recommendations**: Key findings and actionable recommendations based on the data.
- **Fallback to Mock Data**: Automatically generates realistic mock data if actual data files are unavailable.

---

## Prerequisites
To run this app, you need the following installed on your system:
- Python 3.8 or higher
- Pip (Python package manager)
- Git (optional, for cloning the repository)

---

## Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-repo/solar-farm-dashboard.git
   cd solar-farm-dashboard
   ```

2. **Set Up a Virtual Environment** (Optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare Data Files** (Optional):
   - Place cleaned CSV files (`benin_clean.csv`, `sierraleone_clean.csv`, `togo_clean.csv`) in the `data/` folder.
   - Ensure the `data/` folder is added to `.gitignore` to avoid committing large files.

---

## Folder Structure
The project folder structure is organized as follows:
```
solar-farm-dashboard/
├── app/
│   ├── __init__.py          # Makes the folder a Python package
│   ├── main.py              # Main Streamlit application script
│   ├── utils.py             # Utility functions for data processing and visualization
├── data/                    # Folder for cleaned CSV files (ignored by Git)
├── .gitignore               # Ignores unnecessary files (e.g., data/, venv/)
├── README.md                # Documentation for the app setup and usage
├── requirements.txt         # List of Python dependencies
└── dashboard_screenshots/   # Screenshots of the deployed dashboard
```

---

## Running the App
1. **Start the Streamlit App**:
   ```bash
   streamlit run app/main.py
   ```
   - The app will launch in your default web browser at `http://localhost:8501`.

2. **Interact with the Dashboard**:
   - Use the sidebar to select countries, metrics, and advanced filters.
   - Explore visualizations and insights.

3. **Fallback to Mock Data**:
   - If the actual data files are missing, the app will automatically generate mock data for demonstration purposes.

---

## Development Process
### Step 1: Environment Setup
- Created a virtual environment to isolate dependencies.
- Installed required libraries using `requirements.txt`.
- Added `.gitignore` to exclude unnecessary files (e.g., `data/`, `venv/`).

### Step 2: Data Preparation
- Cleaned and processed raw solar farm data for Benin, Sierra Leone, and Togo.
- Exported cleaned datasets to `data/benin_clean.csv`, `data/sierraleone_clean.csv`, and `data/togo_clean.csv`.
- Implemented a fallback mechanism to generate realistic mock data if actual data files are unavailable.

### Step 3: Streamlit App Development
- Designed the app layout with a focus on usability and interactivity.
- Integrated utility functions (`utils.py`) for data loading, filtering, and processing.
- Added interactive widgets (e.g., multiselect, sliders) to allow users to customize their analysis.
- Created visualizations using Plotly and Matplotlib.
- Displayed key insights and actionable recommendations based on the data.

### Step 4: Testing and Debugging
- Tested the app locally to ensure all features worked as expected.
- Handled edge cases (e.g., missing data, invalid filters).
- Ensured compatibility with both actual and mock data.

### Step 5: Deployment
- Deployed the app to **Streamlit Community Cloud** for public access.
- Documented deployment steps and provided a public URL for the dashboard.

---

## Key Performance Indicators (KPIs)
- **Dashboard Usability**: Intuitive navigation with clear labels and user-friendly widgets.
- **Interactive Elements**: Effective use of Streamlit widgets to enhance user engagement.
- **Visual Appeal**: Clean and professional design that effectively communicates data insights.
- **Deployment Success**: Fully functional deployment, accessible via a public URL.

---

## Deployment
### Option 1: Streamlit Community Cloud
1. Push your code to a GitHub repository.
2. Sign up for a free account on [Streamlit Community Cloud](https://streamlit.io/cloud).
3. Create a new app and connect it to your GitHub repository.
4. Specify the main script (`app/main.py`) and Python dependencies (`requirements.txt`).
5. Deploy the app and share the public URL.

### Option 2: Local Deployment
Run the app locally using the command:
```bash
streamlit run app/main.py
```

---

## Contributing
Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a new branch for your feature or bug fix:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. Commit your changes:
   ```bash
   git commit -m "feat: add your feature description"
   ```
4. Push to the branch:
   ```bash
   git push origin feature/your-feature-name
   ```
5. Open a pull request detailing your changes.

---

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgments
- Special thanks to the 10 Academy team for providing the dataset and challenge framework.
- Inspired by best practices in data visualization and dashboard development.

