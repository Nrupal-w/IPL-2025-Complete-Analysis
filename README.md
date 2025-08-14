
#  IPL 2025 Complete Analysis

##  Overview
This project analyzes the **IPL 2025 season** using batting and bowling datasets, calculates performance metrics, generates visual insights, and exports a **comprehensive PDF report**.  
It follows an **industrial programming pattern** with modular code, reusable functions, and structured documentation.

##  Features
- Data cleaning and preprocessing for batting & bowling stats
- Calculation of advanced performance metrics (Composite Score, Weighted Score, Impact Score, Bowling Score)
- Top player rankings across multiple categories
- Team-wise batting and bowling performance breakdowns
- Correlation heatmaps for statistical relationships
- Automatic PDF report generation with professional visualizations
- Industrial-standard comments and function headers

## 🛠 Technologies Used
- **Python 3**
- **Pandas** – Data manipulation
- **NumPy** – Numerical processing
- **Matplotlib & Seaborn** – Data visualization
- **scikit-learn** – Feature scaling
- **PdfPages** – Multi-page PDF export

##  Dataset Information
Two CSV files are used:
- `IPL2025Batters.csv` – Batting statistics
- `IPL2025Bowlers.csv` – Bowling statistics

##  Workflow
1. Load batting and bowling datasets.
2. Clean and prepare data (handle missing values, convert data types).
3. Calculate batting and bowling scores.
4. Identify top performers and generate visualizations.
5. Compile results into a single PDF report.
6. Print key statistics in the console.

##  How to Run
```bash
# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn

# Run the script
python ipl_analysis.py
```

##  Outputs
- **PDF Report:** `IPL2025_Complete_Analysis_Report.pdf`
- **Console Summary:** Top performers and team summaries
- **Preview Visualizations:**  

![Top Runs](/mnt/data/ipltmp_images/plot_preview_1.png)
![Top Strike Rates](/mnt/data/ipltmp_images/plot_preview_2.png)
![Bowling Stats](/mnt/data/ipltmp_images/plot_preview_3.png)

##  Key Highlights from IPL 2025
- **Most Runs:** Automatically detected from dataset
- **Best Strike Rate:** Minimum 50 balls faced
- **Most Wickets:** Across all bowlers
- **Best Economy Rate:** Minimum 20 overs bowled
- **Team-wise leaders** in batting and bowling

##  File Structure
```
.
├── IPL2025Batters.csv
├── IPL2025Bowlers.csv
├── ipl_analysis.py
├── IPL2025_Complete_Analysis_Report.pdf
└── README.md
```

##  Author
**NRUPAL WAKODE**  
 *14/08/2025*
