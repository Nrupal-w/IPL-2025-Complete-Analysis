########################################################
# Project         :    IPL 2025 Complete Analysis
# File Name       :    ipl_analysis.py
# Description     :    End-to-end pipeline to read, clean, score, analyze,
#                      and visualize IPL 2025 batting & bowling data, and
#                      export a comprehensive PDF report with summary stats.
# Author          :    NRUPAL WAKODE
# Date            :    14/08/2025
########################################################

########################################################
# Imports
# Description     :    This section imports all required Python libraries
#                      for data manipulation, visualization, ML preprocessing,
#                      and multi-page PDF export.
# Author          :    NRUPAL WAKODE
# Date            :    14/08/2025
########################################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.preprocessing import MinMaxScaler
import warnings

# Style & warnings
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


########################################################
# Class Name      :    IPLAnalyzer
# Description     :    Encapsulates data loading, preparation, scoring,
#                      team analysis, visualizations, and console summaries.
# Author          :    NRUPAL WAKODE
# Date            :    14/08/2025
########################################################
class IPLAnalyzer:
    ########################################################
    # Function name :    __init__
    # Description   :    Initialize analyzer by reading CSVs and preparing data
    # Input         :    batter_file (str), bowler_file (str)
    # Output        :    None (sets self.batter_df, self.bowler_df)
    # Author        :    NRUPAL WAKODE
    # Date          :    14/08/2025
    ########################################################
    def __init__(self, batter_file, bowler_file):
        self.batter_df = pd.read_csv(batter_file)
        self.bowler_df = pd.read_csv(bowler_file)
        self.prepare_data()

    ########################################################
    # Function name :    prepare_data
    # Description   :    Clean & cast columns to numeric, fill missing values
    # Input         :    self
    # Output        :    self.batter_df, self.bowler_df (cleaned)
    # Author        :    NRUPAL WAKODE
    # Date          :    14/08/2025
    ########################################################
    def prepare_data(self):
        self.bowler_df.columns = self.bowler_df.columns.str.strip()

        self.batter_df = self.batter_df.fillna(0)
        self.bowler_df = self.bowler_df.fillna(0)

        # Batting numeric conversions
        batter_cols = ['Runs', 'Matches', 'Inn', 'No', 'HS', 'AVG', 'BF', 'SR', '100s', '50s', '4s', '6s']
        for col in batter_cols:
            if col in self.batter_df.columns:
                if col == 'HS':
                    self.batter_df[col] = (
                        self.batter_df[col].astype(str).str.replace('*', '', regex=False)
                        .apply(pd.to_numeric, errors='coerce').fillna(0)
                    )
                else:
                    self.batter_df[col] = pd.to_numeric(self.batter_df[col], errors='coerce').fillna(0)

        # Bowling numeric conversions
        bowler_cols = ['WKT', 'MAT', 'INN', 'OVR', 'RUNS', 'AVG', 'ECO', 'SR', '4W', '5W']
        for col in bowler_cols:
            if col in self.bowler_df.columns:
                self.bowler_df[col] = pd.to_numeric(self.bowler_df[col], errors='coerce').fillna(0)

    ########################################################
    # Function name :    calculate_batting_scores
    # Description   :    Compute composite, weighted (0–10), and impact scores
    # Input         :    self
    # Output        :    self.batter_df with score columns
    # Author        :    NRUPAL WAKODE
    # Date          :    14/08/2025
    ########################################################
    def calculate_batting_scores(self):
        # Simple composite score
        self.batter_df['Composite Score'] = (
            self.batter_df['Runs'] * 0.5 +
            self.batter_df['SR'] * 0.3 +
            self.batter_df['50s'] * 10 +
            self.batter_df['100s'] * 25
        )

        # Weighted normalized score (scaled 0–10)
        batter_cols = ['Runs', 'AVG', 'SR', '100s', '50s', '4s', '6s', 'HS', 'No', 'Matches', 'Inn', 'BF']
        scaler = MinMaxScaler()
        batter_scaled = pd.DataFrame(
            scaler.fit_transform(self.batter_df[batter_cols]),
            columns=batter_cols, index=self.batter_df.index
        )

        batter_scaled['Weighted Score'] = (
            batter_scaled['Runs'] * 0.25 +
            batter_scaled['AVG'] * 0.20 +
            batter_scaled['SR'] * 0.15 +
            batter_scaled['100s'] * 0.10 +
            batter_scaled['50s'] * 0.08 +
            batter_scaled['6s'] * 0.07 +
            batter_scaled['4s'] * 0.05 +
            batter_scaled['HS'] * 0.04 +
            batter_scaled['No'] * 0.03 +
            batter_scaled['Matches'] * 0.02 +
            batter_scaled['Inn'] * 0.01 +
            batter_scaled['BF'] * 0.005
        ) * 10

        self.batter_df['Weighted Score'] = batter_scaled['Weighted Score']

        # Impact score
        self.batter_df['Impact Score'] = (
            (self.batter_df['6s'] * 6 + self.batter_df['4s'] * 4) /
            self.batter_df['BF'].replace(0, 1) * 100 +
            self.batter_df['SR'] * 0.5
        )
        return self.batter_df

    ########################################################
    # Function name :    calculate_bowling_scores
    # Description   :    Compute bowling score (0–10), economy perf, wicket score
    # Input         :    self
    # Output        :    self.bowler_df with score columns
    # Author        :    NRUPAL WAKODE
    # Date          :    14/08/2025
    ########################################################
    def calculate_bowling_scores(self):
        bowler_cols = ['WKT', 'AVG', 'ECO', 'SR', '4W', '5W', 'MAT', 'INN', 'OVR', 'RUNS']
        scaler = MinMaxScaler()
        bowler_scaled = pd.DataFrame(
            scaler.fit_transform(self.bowler_df[bowler_cols]),
            columns=bowler_cols, index=self.bowler_df.index
        )

        bowler_scaled['Bowling Score'] = (
            bowler_scaled['WKT'] * 0.25 +
            (1 - bowler_scaled['ECO']) * 0.20 +
            (1 - bowler_scaled['AVG']) * 0.15 +
            (1 - bowler_scaled['SR']) * 0.15 +
            bowler_scaled['4W'] * 0.08 +
            bowler_scaled['5W'] * 0.12 +
            bowler_scaled['OVR'] * 0.05
        ) * 10

        self.bowler_df['Bowling Score'] = bowler_scaled['Bowling Score']

        # Economy performance (scaled 0–10, higher is better)
        eco_min = self.bowler_df['ECO'].min()
        eco_max = self.bowler_df['ECO'].max()
        denom = (eco_max - eco_min) if (eco_max - eco_min) != 0 else 1
        self.bowler_df['Economy Performance'] = 10 - (self.bowler_df['ECO'] - eco_min) / denom * 10

        # Wicket-taking score
        self.bowler_df['Wicket Score'] = (
            self.bowler_df['WKT'] * 2 +
            self.bowler_df['4W'] * 8 +
            self.bowler_df['5W'] * 15
        )
        return self.bowler_df

    ########################################################
    # Function name :    get_top_performers
    # Description   :    Select top-N performers across key metrics
    # Input         :    self, n (int)
    # Output        :    dict of DataFrames
    # Author        :    NRUPAL WAKODE
    # Date          :    14/08/2025
    ########################################################
    def get_top_performers(self, n=10):
        return {
            'runs': self.batter_df.nlargest(n, 'Runs'),
            'strike_rate': self.batter_df[self.batter_df['BF'] >= 50].nlargest(n, 'SR'),
            'composite': self.batter_df.nlargest(n, 'Composite Score'),
            'weighted': self.batter_df.nlargest(n, 'Weighted Score'),
            'impact': self.batter_df.nlargest(n, 'Impact Score'),
            'wickets': self.bowler_df.nlargest(n, 'WKT'),
            'economy': self.bowler_df[self.bowler_df['OVR'] >= 20].nsmallest(n, 'ECO'),
            'bowling_score': self.bowler_df.nlargest(n, 'Bowling Score'),
            'wicket_takers': self.bowler_df.nlargest(n, 'Wicket Score')
        }

    ########################################################
    # Function name :    team_analysis
    # Description   :    Aggregate batting & bowling performance by team
    # Input         :    self
    # Output        :    team_batting (DataFrame), team_bowling (DataFrame)
    # Author        :    NRUPAL WAKODE
    # Date          :    14/08/2025
    ########################################################
    def team_analysis(self):
        team_batting = self.batter_df.groupby('Team').agg({
            'Runs': 'sum',
            'SR': 'mean',
            'AVG': 'mean',
            '100s': 'sum',
            '50s': 'sum',
            '6s': 'sum',
            '4s': 'sum',
            'Weighted Score': 'mean'
        }).round(2)

        team_bowling = self.bowler_df.groupby('Team').agg({
            'WKT': 'sum',
            'ECO': 'mean',
            'AVG': 'mean',
            'SR': 'mean',
            '4W': 'sum',
            '5W': 'sum',
            'Bowling Score': 'mean'
        }).round(2)

        return team_batting, team_bowling

    ########################################################
    # Function name :    create_visualizations
    # Description   :    Generate all plots and export a multi-page PDF report
    # Input         :    self, output_file (str)
    # Output        :    output_file (str)
    # Author        :    NRUPAL WAKODE
    # Date          :    14/08/2025
    ########################################################
    def create_visualizations(self, output_file='IPL2025_Complete_Analysis_Report.pdf'):
        top_performers = self.get_top_performers(10)
        team_batting, team_bowling = self.team_analysis()

        with PdfPages(output_file) as pdf:

            ########################################################
            # Function name :    plot_top_run_scorers
            # Description   :    Plots top 15 run scorers in IPL 2025
            # Input         :    batter_df, pdf
            # Output        :    Bar chart saved to PDF
            # Author        :    NRUPAL WAKODE
            # Date          :    14/08/2025
            ########################################################
            plt.figure(figsize=(14, 8))
            top_runs = top_performers['runs'].head(15)
            bars = plt.bar(range(len(top_runs)), top_runs['Runs'],
                           color=plt.cm.viridis(np.linspace(0, 1, len(top_runs))))
            plt.title('Top 15 Run Scorers - IPL 2025', fontsize=16, fontweight='bold')
            plt.xlabel('Players', fontsize=12)
            plt.ylabel('Runs', fontsize=12)
            plt.xticks(range(len(top_runs)), top_runs['Player Name'], rotation=45, ha='right')
            for i, bar in enumerate(bars):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                         str(int(top_runs.iloc[i]['Runs'])), ha='center', va='bottom', fontsize=9)
            plt.tight_layout()
            pdf.savefig(bbox_inches='tight')
            plt.close()

            ########################################################
            # Function name :    plot_best_strike_rates
            # Description   :    Plots top 15 strike rates (minimum 50 balls faced)
            # Input         :    batter_df, pdf
            # Output        :    Bar chart saved to PDF
            # Author        :    NRUPAL WAKODE
            # Date          :    14/08/2025
            ########################################################
            plt.figure(figsize=(14, 8))
            top_sr = top_performers['strike_rate'].head(15)
            bars = plt.bar(range(len(top_sr)), top_sr['SR'],
                           color=plt.cm.plasma(np.linspace(0, 1, len(top_sr))))
            plt.title('Top 15 Strike Rates - IPL 2025 (Min 50 balls)', fontsize=16, fontweight='bold')
            plt.xlabel('Players', fontsize=12)
            plt.ylabel('Strike Rate', fontsize=12)
            plt.xticks(range(len(top_sr)), top_sr['Player Name'], rotation=45, ha='right')
            for i, bar in enumerate(bars):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                         f"{top_sr.iloc[i]['SR']:.1f}", ha='center', va='bottom', fontsize=9)
            plt.tight_layout()
            pdf.savefig(bbox_inches='tight')
            plt.close()

            ########################################################
            # Function name :    plot_top_weighted_batting_scores
            # Description   :    Plots top 15 comprehensive batting scores (Weighted Score)
            # Input         :    batter_df, pdf
            # Output        :    Horizontal bar chart saved to PDF
            # Author        :    NRUPAL WAKODE
            # Date          :    14/08/2025
            ########################################################
            plt.figure(figsize=(14, 8))
            top_weighted = top_performers['weighted'].head(15)
            plt.barh(range(len(top_weighted)), top_weighted['Weighted Score'],
                     color=plt.cm.coolwarm(np.linspace(0, 1, len(top_weighted))))
            plt.title('Top 15 Comprehensive Batting Scores - IPL 2025', fontsize=16, fontweight='bold')
            plt.xlabel('Weighted Score (0–10)', fontsize=12)
            plt.ylabel('Players', fontsize=12)
            plt.yticks(range(len(top_weighted)), top_weighted['Player Name'])
            plt.gca().invert_yaxis()
            for i, score in enumerate(top_weighted['Weighted Score']):
                plt.text(score + 0.05, i, f"{score:.2f}", va='center', fontsize=9)
            plt.tight_layout()
            pdf.savefig(bbox_inches='tight')
            plt.close()

            ########################################################
            # Function name :    plot_top_wicket_takers
            # Description   :    Plots top 15 wicket takers in IPL 2025
            # Input         :    bowler_df, pdf
            # Output        :    Bar chart saved to PDF
            # Author        :    NRUPAL WAKODE
            # Date          :    14/08/2025
            ########################################################
            plt.figure(figsize=(14, 8))
            top_wickets = top_performers['wickets'].head(15)
            bars = plt.bar(range(len(top_wickets)), top_wickets['WKT'],
                           color=plt.cm.Set3(np.linspace(0, 1, len(top_wickets))))
            plt.title('Top 15 Wicket Takers - IPL 2025', fontsize=16, fontweight='bold')
            plt.xlabel('Players', fontsize=12)
            plt.ylabel('Wickets', fontsize=12)
            plt.xticks(range(len(top_wickets)), top_wickets['Player Name'], rotation=45, ha='right')
            for i, bar in enumerate(bars):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                         str(int(top_wickets.iloc[i]['WKT'])), ha='center', va='bottom', fontsize=9)
            plt.tight_layout()
            pdf.savefig(bbox_inches='tight')
            plt.close()

            ########################################################
            # Function name :    plot_best_economy_rates
            # Description   :    Plots top 15 economy rates (minimum 20 overs)
            # Input         :    bowler_df, pdf
            # Output        :    Bar chart saved to PDF
            # Author        :    NRUPAL WAKODE
            # Date          :    14/08/2025
            ########################################################
            plt.figure(figsize=(14, 8))
            top_eco = top_performers['economy'].head(15)
            bars = plt.bar(range(len(top_eco)), top_eco['ECO'],
                           color=plt.cm.RdYlGn_r(np.linspace(0, 1, len(top_eco))))
            plt.title('Best Economy Rates - IPL 2025 (Min 20 overs)', fontsize=16, fontweight='bold')
            plt.xlabel('Players', fontsize=12)
            plt.ylabel('Economy Rate', fontsize=12)
            plt.xticks(range(len(top_eco)), top_eco['Player Name'], rotation=45, ha='right')
            for i, bar in enumerate(bars):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                         f"{top_eco.iloc[i]['ECO']:.2f}", ha='center', va='bottom', fontsize=9)
            plt.tight_layout()
            pdf.savefig(bbox_inches='tight')
            plt.close()

            ########################################################
            # Function name :    plot_top_bowling_scores
            # Description   :    Plots top 15 comprehensive bowling scores
            # Input         :    bowler_df, pdf
            # Output        :    Horizontal bar chart saved to PDF
            # Author        :    NRUPAL WAKODE
            # Date          :    14/08/2025
            ########################################################
            plt.figure(figsize=(14, 8))
            top_bowling = top_performers['bowling_score'].head(15)
            plt.barh(range(len(top_bowling)), top_bowling['Bowling Score'],
                     color=plt.cm.viridis(np.linspace(0, 1, len(top_bowling))))
            plt.title('Top 15 Comprehensive Bowling Scores - IPL 2025', fontsize=16, fontweight='bold')
            plt.xlabel('Bowling Score (0–10)', fontsize=12)
            plt.ylabel('Players', fontsize=12)
            plt.yticks(range(len(top_bowling)), top_bowling['Player Name'])
            plt.gca().invert_yaxis()
            for i, score in enumerate(top_bowling['Bowling Score']):
                plt.text(score + 0.05, i, f"{score:.2f}", va='center', fontsize=9)
            plt.tight_layout()
            pdf.savefig(bbox_inches='tight')
            plt.close()

            ########################################################
            # Function name :    plot_team_batting_performance
            # Description   :    Plots team-wise batting metrics (Runs, SR, 6s, 100s, 50s)
            # Input         :    team_batting_df, pdf
            # Output        :    Grouped bar chart saved to PDF
            # Author        :    NRUPAL WAKODE
            # Date          :    14/08/2025
            ########################################################
            plt.figure(figsize=(15, 10))
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

            # Total runs by team
            team_batting_sorted = team_batting.sort_values('Runs', ascending=False)
            ax1.bar(team_batting_sorted.index, team_batting_sorted['Runs'],
                    color=plt.cm.tab10(np.linspace(0, 1, len(team_batting_sorted))))
            ax1.set_title('Total Runs by Team', fontweight='bold')
            ax1.set_ylabel('Runs')
            ax1.tick_params(axis='x', rotation=45)

            # Average strike rate by team
            team_sr_sorted = team_batting.sort_values('SR', ascending=False)
            ax2.bar(team_sr_sorted.index, team_sr_sorted['SR'],
                    color=plt.cm.tab10(np.linspace(0, 1, len(team_sr_sorted))))
            ax2.set_title('Average Strike Rate by Team', fontweight='bold')
            ax2.set_ylabel('Strike Rate')
            ax2.tick_params(axis='x', rotation=45)

            # Sixes by team
            team_6s_sorted = team_batting.sort_values('6s', ascending=False)
            ax3.bar(team_6s_sorted.index, team_6s_sorted['6s'],
                    color=plt.cm.tab10(np.linspace(0, 1, len(team_6s_sorted))))
            ax3.set_title('Total Sixes by Team', fontweight='bold')
            ax3.set_ylabel('Sixes')
            ax3.tick_params(axis='x', rotation=45)

            # Milestones (50s + 100s)
            team_milestones = team_batting['100s'] + team_batting['50s']
            team_milestones_sorted = team_milestones.sort_values(ascending=False)
            ax4.bar(team_milestones_sorted.index, team_milestones_sorted.values,
                    color=plt.cm.tab10(np.linspace(0, 1, len(team_milestones_sorted))))
            ax4.set_title('Total Milestones (50s + 100s) by Team', fontweight='bold')
            ax4.set_ylabel('Milestones')
            ax4.tick_params(axis='x', rotation=45)

            plt.tight_layout()
            pdf.savefig(bbox_inches='tight')
            plt.close()

            ########################################################
            # Function name :    plot_team_bowling_performance
            # Description   :    Plots team-wise bowling metrics (WKT, ECO, 4W, 5W)
            # Input         :    team_bowling_df, pdf
            # Output        :    Grouped bar chart saved to PDF
            # Author        :    NRUPAL WAKODE
            # Date          :    14/08/2025
            ########################################################
            plt.figure(figsize=(15, 10))
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

            # Total wickets by team
            team_bowling_sorted = team_bowling.sort_values('WKT', ascending=False)
            ax1.bar(team_bowling_sorted.index, team_bowling_sorted['WKT'],
                    color=plt.cm.Set1(np.linspace(0, 1, len(team_bowling_sorted))))
            ax1.set_title('Total Wickets by Team', fontweight='bold')
            ax1.set_ylabel('Wickets')
            ax1.tick_params(axis='x', rotation=45)

            # Average economy rate by team
            team_eco_sorted = team_bowling.sort_values('ECO', ascending=True)
            ax2.bar(team_eco_sorted.index, team_eco_sorted['ECO'],
                    color=plt.cm.Set1(np.linspace(0, 1, len(team_eco_sorted))))
            ax2.set_title('Average Economy Rate by Team', fontweight='bold')
            ax2.set_ylabel('Economy Rate')
            ax2.tick_params(axis='x', rotation=45)

            # 4-wicket hauls by team
            team_4w_sorted = team_bowling.sort_values('4W', ascending=False)
            ax3.bar(team_4w_sorted.index, team_4w_sorted['4W'],
                    color=plt.cm.Set1(np.linspace(0, 1, len(team_4w_sorted))))
            ax3.set_title('4-Wicket Hauls by Team', fontweight='bold')
            ax3.set_ylabel('4-Wicket Hauls')
            ax3.tick_params(axis='x', rotation=45)

            # 5-wicket hauls by team
            team_5w_sorted = team_bowling.sort_values('5W', ascending=False)
            ax4.bar(team_5w_sorted.index, team_5w_sorted['5W'],
                    color=plt.cm.Set1(np.linspace(0, 1, len(team_5w_sorted))))
            ax4.set_title('5-Wicket Hauls by Team', fontweight='bold')
            ax4.set_ylabel('5-Wicket Hauls')
            ax4.tick_params(axis='x', rotation=45)

            plt.tight_layout()
            pdf.savefig(bbox_inches='tight')
            plt.close()

            ########################################################
            # Function name :    plot_batting_correlation
            # Description   :    Plots correlation heatmap of batting metrics
            # Input         :    batter_df, pdf
            # Output        :    Heatmap saved to PDF
            # Author        :    NRUPAL WAKODE
            # Date          :    14/08/2025
            ########################################################
            plt.figure(figsize=(12, 8))
            batting_corr_cols = ['Runs', 'AVG', 'SR', '100s', '50s', '4s', '6s', 'Weighted Score']
            correlation_matrix = self.batter_df[batting_corr_cols].corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                        square=True, cbar_kws={'shrink': 0.8})
            plt.title('Batting Statistics Correlation Matrix', fontsize=16, fontweight='bold')
            plt.tight_layout()
            pdf.savefig(bbox_inches='tight')
            plt.close()

            ########################################################
            # Function name :    plot_summary_table
            # Description   :    Renders summary table of key performance highlights
            # Input         :    batter_df, bowler_df, pdf
            # Output        :    Styled table saved to PDF
            # Author        :    NRUPAL WAKODE
            # Date          :    14/08/2025
            ########################################################
            fig, ax = plt.subplots(figsize=(14, 10))
            ax.axis('tight')
            ax.axis('off')

            # Prepare summary data (with safe indexing)
            def safe_idxmax(df, col, cond=None):
                src = df if cond is None else df[cond]
                if len(src) == 0:
                    return None
                return src[col].idxmax()

            hs_idx = safe_idxmax(self.batter_df, 'HS')
            runs_idx = safe_idxmax(self.batter_df, 'Runs')
            sr_filter = self.batter_df['BF'] >= 50
            sr_idx = safe_idxmax(self.batter_df, 'SR', cond=sr_filter)
            sixes_idx = safe_idxmax(self.batter_df, '6s')
            wkt_idx = safe_idxmax(self.bowler_df, 'WKT')
            eco_filter = self.bowler_df['OVR'] >= 20
            eco_idx = self.bowler_df[eco_filter]['ECO'].idxmin() if eco_filter.any() else None
            fourw_idx = safe_idxmax(self.bowler_df, '4W')
            fivew_idx = safe_idxmax(self.bowler_df, '5W')

            summary_rows = []
            summary_rows.append(['Highest Individual Score',
                                 self.batter_df.loc[hs_idx, 'Player Name'] if hs_idx is not None else '-',
                                 f"{int(self.batter_df['HS'].max()) if hs_idx is not None else '-'}"])
            summary_rows.append(['Most Runs',
                                 self.batter_df.loc[runs_idx, 'Player Name'] if runs_idx is not None else '-',
                                 f"{int(self.batter_df['Runs'].max()) if runs_idx is not None else '-'}"])
            summary_rows.append(['Best Strike Rate (Min 50 BF)',
                                 self.batter_df.loc[sr_idx, 'Player Name'] if sr_idx is not None else '-',
                                 f"{self.batter_df[sr_filter]['SR'].max():.2f}" if sr_idx is not None else '-'])
            summary_rows.append(['Most Sixes',
                                 self.batter_df.loc[sixes_idx, 'Player Name'] if sixes_idx is not None else '-',
                                 f"{int(self.batter_df['6s'].max()) if sixes_idx is not None else '-'}"])
            summary_rows.append(['Most Wickets',
                                 self.bowler_df.loc[wkt_idx, 'Player Name'] if wkt_idx is not None else '-',
                                 f"{int(self.bowler_df['WKT'].max()) if wkt_idx is not None else '-'}"])
            summary_rows.append(['Best Economy (Min 20 OVR)',
                                 self.bowler_df.loc[eco_idx, 'Player Name'] if eco_idx is not None else '-',
                                 f"{self.bowler_df[eco_filter]['ECO'].min():.2f}" if eco_idx is not None else '-'])
            summary_rows.append(['Most 4W Hauls',
                                 self.bowler_df.loc[fourw_idx, 'Player Name'] if fourw_idx is not None else '-',
                                 f"{int(self.bowler_df['4W'].max()) if fourw_idx is not None else '-'}"])
            summary_rows.append(['Most 5W Hauls',
                                 self.bowler_df.loc[fivew_idx, 'Player Name'] if fivew_idx is not None else '-',
                                 f"{int(self.bowler_df['5W'].max()) if fivew_idx is not None else '-'}"])

            summary_df = pd.DataFrame(summary_rows, columns=['Category', 'Player', 'Value'])

            table = ax.table(cellText=summary_df.values,
                             colLabels=summary_df.columns,
                             cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
            table.auto_set_font_size(False)
            table.set_fontsize(12)
            table.scale(1.2, 2)

            # Header styling
            for i in range(len(summary_df.columns)):
                table[(0, i)].set_facecolor('#4CAF50')
                table[(0, i)].set_text_props(weight='bold', color='white')

            plt.title('IPL 2025 - Key Performance Highlights', fontsize=16, fontweight='bold', pad=20)
            pdf.savefig(bbox_inches='tight')
            plt.close()

        print(f"✅ Complete analysis report saved as '{output_file}'")
        return output_file

    ########################################################
    # Function name :    print_summary_stats
    # Description   :    Print key top-performer lists and team summaries
    # Input         :    self
    # Output        :    Console printout
    # Author        :    NRUPAL WAKODE
    # Date          :    14/08/2025
    ########################################################
    def print_summary_stats(self):
        print("=" * 80)
        print("                    IPL 2025 ANALYSIS SUMMARY")
        print("=" * 80)

        print("\n TOP BATTING PERFORMANCES:")
        print("-" * 40)
        top_runs = self.batter_df.nlargest(5, 'Runs')[['Player Name', 'Team', 'Runs', 'AVG', 'SR']]
        print(top_runs.to_string(index=False))

        print("\n TOP BOWLING PERFORMANCES:")
        print("-" * 40)
        top_wickets = self.bowler_df.nlargest(5, 'WKT')[['Player Name', 'Team', 'WKT', 'AVG', 'ECO']]
        print(top_wickets.to_string(index=False))

        print("\n TEAM STATISTICS:")
        print("-" * 40)
        team_batting, team_bowling = self.team_analysis()

        print("\nTeam Batting Summary:")
        team_summary = team_batting[['Runs', 'SR', '6s', '100s', '50s']].round(2)
        print(team_summary.to_string())

        print("\nTeam Bowling Summary:")
        bowling_summary = team_bowling[['WKT', 'ECO', '4W', '5W']].round(2)
        print(bowling_summary.to_string())

        print("\n" + "=" * 80)


########################################################
# Function name :    main
# Description   :    Drive the pipeline: compute scores, visualize, summarize
# Input         :    None (uses hardcoded CSV paths below)
# Output        :    PDF report + console outputs
# Author        :    NRUPAL WAKODE
# Date          :    14/08/2025
########################################################
def main():
    print(" Starting IPL 2025 Complete Analysis...")

    # File names (adjust if needed)
    batter_file = "IPL2025Batters.csv"
    bowler_file = "IPL2025Bowlers.csv"

    analyzer = IPLAnalyzer(batter_file, bowler_file)

    print("Calculating batting scores...")
    analyzer.calculate_batting_scores()

    print("Calculating bowling scores...")
    analyzer.calculate_bowling_scores()

    print("Creating comprehensive visualizations...")
    report_file = analyzer.create_visualizations()

    analyzer.print_summary_stats()

    print(f"\n Analysis complete! Report saved as '{report_file}'")
    print("The report includes:")
    print(" Top run scorers and strike rates")
    print(" Best bowling performances")
    print(" Team-wise analysis")
    print(" Comprehensive scoring systems")
    print(" Statistical correlations")
    print(" Key performance highlights")


########################################################
# Entrypoint
########################################################
if __name__ == "__main__":
    main()
