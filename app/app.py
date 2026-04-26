"""
================================================================================
STREAMLIT-BASED INTERACTIVE DECISION SUPPORT SYSTEM
================================================================================

Project: District-Level Multidimensional Socioeconomic Deprivation Analysis
         in Karnataka, India

Purpose: Web-based interactive platform for policy-oriented decision support
         using rule-based analysis (NOT machine learning)

Author: [Your Name]
Date: February 2026
Version: 1.0 (Web Interactive)

FEATURES:
- District selection and profiling
- Real-time scenario analysis (what-if simulations)
- Policy recommendations based on expert rules
- Spatial cluster integration (LISA)
- Interactive visualizations
- Downloadable reports

================================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from io import BytesIO
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Karnataka Deprivation DSS",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'


# ================================================================================
# BACKEND FUNCTIONS (Rule-Based DSS Logic)
# ================================================================================

class DeprivationDSS:
    """
    Rule-based Decision Support System for deprivation analysis.
    Uses expert-defined thresholds and policy frameworks.
    """
    
    def __init__(self, data_path: str):
        """
        Initialize DSS with processed data.
        
        Parameters:
        -----------
        data_path : str
            Path to module1_processed_data.csv
        """
        self.data = pd.read_csv(data_path)
        self.lisa_data = None
        self._load_lisa_data()
        self._ensure_required_columns()
        
    def _load_lisa_data(self):
        """Load LISA results if available."""
        try:
            BASE_DIR = Path(__file__).resolve().parent
            data_path1 = BASE_DIR / "data" / "module4_lisa_results.csv"
            
            lisa_path = Path(data_path1)
            if lisa_path.exists():
                self.lisa_data = pd.read_csv(lisa_path)
                # Merge LISA data
                self.data = self.data.merge(
                    self.lisa_data[['District', 'Cluster_Type']],
                    on='District',
                    how='left'
                )
        except Exception as e:
            st.warning(f"LISA data not available: {e}")
    
    def _ensure_required_columns(self):
        """Ensure all required columns exist."""
        # Check for SEDI column (might be SEDI_Score or SEDI)
        if 'SEDI' not in self.data.columns and 'SEDI_Score' in self.data.columns:
            self.data['SEDI'] = self.data['SEDI_Score']
        elif 'SEDI_Score' not in self.data.columns and 'SEDI' in self.data.columns:
            self.data['SEDI_Score'] = self.data['SEDI']
        
        # Ensure Deprivation_Category exists
        if 'Deprivation_Category' not in self.data.columns:
            self.data = self._categorize_deprivation(self.data)
    
    def _categorize_deprivation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Categorize deprivation using percentile method."""
        p33 = df['SEDI'].quantile(0.33)
        p67 = df['SEDI'].quantile(0.67)
        
        def assign_category(sedi):
            if sedi < p33:
                return 'High Deprivation'
            elif sedi < p67:
                return 'Medium Deprivation'
            else:
                return 'Low Deprivation'
        
        df['Deprivation_Category'] = df['SEDI'].apply(assign_category)
        return df
    
    def get_district_profile(self, district_name: str) -> dict:
        """
        Get comprehensive profile for a district.
        
        Parameters:
        -----------
        district_name : str
            District name
        
        Returns:
        --------
        dict
            District profile with all indicators and assessments
        """
        district_data = self.data[self.data['District'] == district_name].iloc[0]
        
        profile = {
            'district': district_name,
            'sedi': float(district_data['SEDI']),
            'category': district_data['Deprivation_Category'],
            'rank': int(self.data['SEDI'].rank(ascending=False)[district_data.name]) if 'SEDI' in self.data.columns else None,
            'total_districts': len(self.data),
            'cluster_type': district_data.get('Cluster_Type', 'N/A'),
            'indicators': {
                'Literacy Rate (%)': float(district_data.get('Literacy_Rate', 0)),
                'Per Capita Income (₹)': float(district_data.get('Per_Capita_Income', 0)),
                'Unemployment Rate (%)': float(district_data.get('Unemployment_Rate', 0)),
                'Healthcare Facilities (per lakh)': float(district_data.get('Healthcare_Facilities_Per_Lakh', 0)),
                'Electricity Access (%)': float(district_data.get('Electricity_Access', 0)),
                'Safe Water Access (%)': float(district_data.get('Safe_Water_Access', 0)),
                'Road Density (km/sq.km)': float(district_data.get('Road_Density', 0)),
                'Urbanization Rate (%)': float(district_data.get('Urbanization_Rate', 0))
            }
        }
        
        return profile
    
    def diagnose_sectoral_deficits(self, district_name: str) -> dict:
        """
        Diagnose sector-specific deficits for a district.
        
        Parameters:
        -----------
        district_name : str
            District name
        
        Returns:
        --------
        dict
            Sectoral deficit assessment
        """
        district_data = self.data[self.data['District'] == district_name].iloc[0]
        
        # Define thresholds (expert-defined)
        thresholds = {
            'Economic': {
                'indicators': ['Per_Capita_Income', 'Unemployment_Rate'],
                'Per_Capita_Income': self.data['Per_Capita_Income'].quantile(0.25),
                'Unemployment_Rate': self.data['Unemployment_Rate'].quantile(0.75)
            },
            'Education': {
                'indicators': ['Literacy_Rate'],
                'Literacy_Rate': 75.0  # National average target
            },
            'Health': {
                'indicators': ['Healthcare_Facilities_Per_Lakh'],
                'Healthcare_Facilities_Per_Lakh': self.data['Healthcare_Facilities_Per_Lakh'].quantile(0.33)
            },
            'Infrastructure': {
                'indicators': ['Road_Density', 'Electricity_Access', 'Urbanization_Rate'],
                'Road_Density': self.data['Road_Density'].quantile(0.33),
                'Electricity_Access': 90.0,
                'Urbanization_Rate': self.data['Urbanization_Rate'].quantile(0.33)
            }
        }
        
        deficits = {}
        
        for sector, config in thresholds.items():
            has_deficit = False
            deficit_indicators = []
            
            for indicator in config['indicators']:
                if indicator in district_data.index:
                    threshold = config[indicator]
                    value = district_data[indicator]
                    
                    # Check deficit (lower is worse except for unemployment)
                    if indicator == 'Unemployment_Rate':
                        if value > threshold:
                            has_deficit = True
                            deficit_indicators.append(indicator)
                    else:
                        if value < threshold:
                            has_deficit = True
                            deficit_indicators.append(indicator)
            
            deficits[sector] = {
                'has_deficit': has_deficit,
                'indicators': deficit_indicators,
                'severity': 'High' if len(deficit_indicators) >= 2 else 'Medium' if has_deficit else 'Low'
            }
        
        return deficits
    
    def generate_policy_recommendations(self, district_name: str) -> dict:
        """
        Generate rule-based policy recommendations.
        
        Parameters:
        -----------
        district_name : str
            District name
        
        Returns:
        --------
        dict
            Policy recommendations by sector
        """
        profile = self.get_district_profile(district_name)
        deficits = self.diagnose_sectoral_deficits(district_name)
        
        recommendations = {
            'priority': self._determine_priority(profile['category'], deficits),
            'sectors': {}
        }
        
        # Sector-specific recommendations
        sector_programs = {
            'Economic': [
                'Skill Development and Vocational Training Programs',
                'MSME Support and Entrepreneurship Schemes',
                'Rural Employment Guarantee Programs',
                'Agricultural Modernization and Market Linkages',
                'Industrial Corridor Development'
            ],
            'Education': [
                'Adult Literacy and Continuing Education Programs',
                'School Infrastructure Modernization',
                'Digital Education and ICT Integration',
                'Teacher Training and Capacity Building',
                'Scholarship and Incentive Schemes for Disadvantaged Groups'
            ],
            'Health': [
                'Primary Health Center Expansion and Upgradation',
                'Mobile Health Units for Remote Areas',
                'Telemedicine and Digital Health Services',
                'Maternal and Child Health Programs',
                'Disease Prevention and Health Awareness Campaigns'
            ],
            'Infrastructure': [
                'Rural Road Connectivity Projects (PMGSY)',
                'Electricity Grid Extension and Solar Power',
                'Urban Infrastructure Development (Smart Cities)',
                'Water Supply and Sanitation Programs',
                'Telecommunications and Digital Infrastructure'
            ]
        }
        
        for sector, deficit_info in deficits.items():
            if deficit_info['has_deficit']:
                recommendations['sectors'][sector] = {
                    'severity': deficit_info['severity'],
                    'programs': sector_programs[sector][:3],  # Top 3 programs
                    'budget_priority': 'High' if deficit_info['severity'] == 'High' else 'Medium'
                }
        
        # Add spatial context if available
        if profile['cluster_type'] != 'N/A':
            recommendations['spatial_context'] = self._spatial_recommendations(profile['cluster_type'])
        
        return recommendations
    
    def _determine_priority(self, category: str, deficits: dict) -> str:
        """Determine overall intervention priority."""
        deficit_count = sum(1 for d in deficits.values() if d['has_deficit'])
        
        if category == 'High Deprivation' or deficit_count >= 3:
            return 'URGENT'
        elif category == 'Medium Deprivation' or deficit_count >= 2:
            return 'HIGH'
        else:
            return 'MODERATE'
    
    def _spatial_recommendations(self, cluster_type: str) -> str:
        """Generate spatial context recommendations."""
        spatial_rec = {
            'LL': 'HIGH PRIORITY HOTSPOT: This district is in a high deprivation cluster. Implement regional development programs coordinating with neighboring districts for maximum impact through spillover effects.',
            'HH': 'BEST PRACTICE ZONE: Learn from successful policies in this low deprivation cluster. Document and share best practices with other regions.',
            'HL': 'POSITIVE OUTLIER: Despite challenging neighborhood, this district performs well. Investigate success factors for replication.',
            'LH': 'NEGATIVE OUTLIER: Underperforming despite favorable neighborhood. Diagnose local barriers and governance issues.',
            'NS': 'No significant spatial clustering detected. District-specific interventions appropriate.'
        }
        
        return spatial_rec.get(cluster_type, 'Spatial data not available')
    
    def simulate_scenario(self, district_name: str, improvements: dict) -> dict:
        """
        Simulate what-if scenario with sector improvements.
        
        Parameters:
        -----------
        district_name : str
            District name
        improvements : dict
            Sector improvement percentages
            Example: {'education': 10, 'health': 15, 'infrastructure': 5}
        
        Returns:
        --------
        dict
            Scenario analysis results
        """
        district_data = self.data[self.data['District'] == district_name].iloc[0].copy()
        
        # Apply improvements to relevant indicators
        improvement_mapping = {
            'education': ['Literacy_Rate'],
            'health': ['Healthcare_Facilities_Per_Lakh'],
            'infrastructure': ['Road_Density', 'Electricity_Access', 'Urbanization_Rate'],
            'economic': ['Per_Capita_Income']
        }
        
        improved_data = district_data.copy()
        
        for sector, pct_improvement in improvements.items():
            if pct_improvement > 0:
                for indicator in improvement_mapping.get(sector, []):
                    if indicator in improved_data.index:
                        current_value = improved_data[indicator]
                        # Apply percentage improvement
                        improved_value = current_value * (1 + pct_improvement / 100)
                        
                        # Cap at reasonable maximums
                        if indicator in ['Literacy_Rate', 'Electricity_Access']:
                            improved_value = min(improved_value, 100)
                        elif indicator == 'Urbanization_Rate':
                            improved_value = min(improved_value, 100)
                        
                        improved_data[indicator] = improved_value
        
        # Recalculate SEDI (simplified - proportional to indicator improvements)
        # This is a simplified simulation; actual SEDI would need full recalculation
        current_sedi = district_data['SEDI']
        
        # Estimate SEDI improvement based on average improvement
        avg_improvement = np.mean([v for v in improvements.values() if v > 0]) if improvements else 0
        estimated_sedi = current_sedi * (1 + avg_improvement / 200)  # Half effect
        estimated_sedi = min(estimated_sedi, 100)
        
        # Determine new category
        p33 = self.data['SEDI'].quantile(0.33)
        p67 = self.data['SEDI'].quantile(0.67)
        
        if estimated_sedi < p33:
            new_category = 'High Deprivation'
        elif estimated_sedi < p67:
            new_category = 'Medium Deprivation'
        else:
            new_category = 'Low Deprivation'
        
        return {
            'current_sedi': float(current_sedi),
            'estimated_sedi': float(estimated_sedi),
            'sedi_change': float(estimated_sedi - current_sedi),
            'current_category': district_data['Deprivation_Category'],
            'estimated_category': new_category,
            'improvements_applied': improvements,
            'improved_indicators': {
                indicator: {
                    'current': float(district_data[indicator]),
                    'improved': float(improved_data[indicator]),
                    'change_pct': float((improved_data[indicator] - district_data[indicator]) / district_data[indicator] * 100)
                }
                for sector, pct in improvements.items() if pct > 0
                for indicator in improvement_mapping.get(sector, [])
                if indicator in district_data.index
            }
        }
    
    def get_all_districts(self) -> list:
        """Get list of all districts."""
        return sorted(self.data['District'].tolist())
    
    def get_state_summary(self) -> dict:
        """Get state-level summary statistics."""
        return {
            'total_districts': len(self.data),
            'avg_sedi': float(self.data['SEDI'].mean()),
            'category_distribution': self.data['Deprivation_Category'].value_counts().to_dict(),
            'cluster_distribution': self.data['Cluster_Type'].value_counts().to_dict() if 'Cluster_Type' in self.data.columns else {}
        }


# ================================================================================
# VISUALIZATION FUNCTIONS
# ================================================================================

def plot_district_indicators(profile: dict):
    """Create radar/bar chart of district indicators."""
    indicators = profile['indicators']
    
    # Normalize indicators to 0-100 scale for comparison
    normalized = {}
    for name, value in indicators.items():
        if 'Rate' in name or 'Access' in name:
            normalized[name] = value  # Already 0-100
        elif 'Income' in name:
            normalized[name] = min((value / 500000) * 100, 100)  # Scale income
        elif 'Healthcare' in name:
            normalized[name] = min((value / 50) * 100, 100)  # Scale facilities
        elif 'Density' in name:
            normalized[name] = min((value / 5) * 100, 100)  # Scale density
        else:
            normalized[name] = value
    
    # Create bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    
    names = list(normalized.keys())
    values = list(normalized.values())
    
    colors = ['#2ecc71' if v >= 70 else '#f39c12' if v >= 40 else '#e74c3c' for v in values]
    
    bars = ax.barh(names, values, color=colors, alpha=0.7)
    ax.set_xlabel('Normalized Score (0-100)', fontsize=12, fontweight='bold')
    ax.set_title(f'Indicator Profile: {profile["district"]}', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 100)
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars, values):
        width = bar.get_width()
        ax.text(width + 2, bar.get_y() + bar.get_height()/2, 
               f'{val:.1f}', ha='left', va='center', fontweight='bold')
    
    plt.tight_layout()
    return fig


def plot_scenario_comparison(scenario_results: dict):
    """Create comparison plot for scenario analysis."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # SEDI comparison
    sedi_data = {
        'Current': scenario_results['current_sedi'],
        'Projected': scenario_results['estimated_sedi']
    }
    
    colors = ['#3498db', '#2ecc71']
    bars1 = ax1.bar(sedi_data.keys(), sedi_data.values(), color=colors, alpha=0.7)
    ax1.set_ylabel('SEDI Score', fontsize=12, fontweight='bold')
    ax1.set_title('SEDI Score Comparison', fontsize=13, fontweight='bold')
    ax1.set_ylim(0, 100)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars1, sedi_data.values()):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, height + 2,
                f'{val:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # Add change annotation
    change = scenario_results['sedi_change']
    ax1.text(0.5, 0.95, f'Change: +{change:.2f}', 
            transform=ax1.transAxes, ha='center', va='top',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3),
            fontsize=11, fontweight='bold')
    
    # Indicator improvements
    if scenario_results['improved_indicators']:
        indicators = list(scenario_results['improved_indicators'].keys())
        changes = [scenario_results['improved_indicators'][ind]['change_pct'] for ind in indicators]
        
        bars2 = ax2.barh(indicators, changes, color='#27ae60', alpha=0.7)
        ax2.set_xlabel('Improvement (%)', fontsize=12, fontweight='bold')
        ax2.set_title('Indicator Improvements', fontsize=13, fontweight='bold')
        ax2.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for bar, val in zip(bars2, changes):
            width = bar.get_width()
            ax2.text(width + 0.5, bar.get_y() + bar.get_height()/2,
                    f'+{val:.1f}%', ha='left', va='center', fontweight='bold')
    else:
        ax2.text(0.5, 0.5, 'No improvements applied',
                transform=ax2.transAxes, ha='center', va='center',
                fontsize=12, style='italic')
        ax2.set_xticks([])
        ax2.set_yticks([])
    
    plt.tight_layout()
    return fig


def plot_state_overview(dss: DeprivationDSS):
    """Create state-level overview visualization."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Category distribution
    category_counts = dss.data['Deprivation_Category'].value_counts()
    colors_cat = {'Low Deprivation': '#2ecc71', 'Medium Deprivation': '#f39c12', 
                  'High Deprivation': '#e74c3c'}
    pie_colors = [colors_cat.get(cat, '#95a5a6') for cat in category_counts.index]
    
    ax1.pie(category_counts.values, labels=category_counts.index, autopct='%1.1f%%',
           colors=pie_colors, startangle=90)
    ax1.set_title('Deprivation Category Distribution\n(31 Karnataka Districts)', 
                 fontsize=13, fontweight='bold')
    
    # SEDI distribution
    ax2.hist(dss.data['SEDI'], bins=15, color='steelblue', edgecolor='black', alpha=0.7)
    ax2.axvline(dss.data['SEDI'].mean(), color='red', linestyle='--', linewidth=2,
               label=f"Mean: {dss.data['SEDI'].mean():.2f}")
    ax2.set_xlabel('SEDI Score', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Number of Districts', fontsize=12, fontweight='bold')
    ax2.set_title('SEDI Score Distribution', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    return fig


# ================================================================================
# REPORT GENERATION
# ================================================================================

def generate_district_report(dss: DeprivationDSS, district_name: str, 
                            recommendations: dict, scenario: dict = None) -> str:
    """Generate comprehensive text report for a district."""
    profile = dss.get_district_profile(district_name)
    deficits = dss.diagnose_sectoral_deficits(district_name)
    
    report = []
    report.append("="*80)
    report.append(f"DISTRICT DEPRIVATION REPORT: {district_name.upper()}")
    report.append("="*80)
    report.append("")
    report.append("Project: District-Level Socioeconomic Deprivation Analysis - Karnataka")
    report.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # 1. Executive Summary
    report.append("1. EXECUTIVE SUMMARY")
    report.append("-" * 80)
    report.append(f"SEDI Score:              {profile['sedi']:.2f}/100")
    report.append(f"Deprivation Category:    {profile['category']}")
    report.append(f"State Rank:              {profile['rank']}/{profile['total_districts']}")
    report.append(f"Intervention Priority:   {recommendations['priority']}")
    if profile['cluster_type'] != 'N/A':
        report.append(f"Spatial Cluster Type:    {profile['cluster_type']}")
    report.append("")
    
    # 2. Indicator Profile
    report.append("2. INDICATOR PROFILE")
    report.append("-" * 80)
    for indicator, value in profile['indicators'].items():
        report.append(f"{indicator:40s}: {value:>12,.2f}")
    report.append("")
    
    # 3. Sectoral Deficit Analysis
    report.append("3. SECTORAL DEFICIT ANALYSIS")
    report.append("-" * 80)
    for sector, deficit_info in deficits.items():
        status = "DEFICIT" if deficit_info['has_deficit'] else "ADEQUATE"
        severity = deficit_info['severity']
        report.append(f"{sector:20s}: {status:10s} (Severity: {severity})")
        if deficit_info['indicators']:
            report.append(f"  Deficit Indicators: {', '.join(deficit_info['indicators'])}")
    report.append("")
    
    # 4. Policy Recommendations
    report.append("4. POLICY RECOMMENDATIONS")
    report.append("-" * 80)
    report.append(f"Overall Priority: {recommendations['priority']}")
    report.append("")
    
    for sector, rec_info in recommendations['sectors'].items():
        report.append(f"{sector.upper()}:")
        report.append(f"  Severity: {rec_info['severity']}")
        report.append(f"  Budget Priority: {rec_info['budget_priority']}")
        report.append("  Recommended Programs:")
        for i, program in enumerate(rec_info['programs'], 1):
            report.append(f"    {i}. {program}")
        report.append("")
    
    if 'spatial_context' in recommendations:
        report.append("5. SPATIAL CONTEXT AND REGIONAL COORDINATION")
        report.append("-" * 80)
        report.append(recommendations['spatial_context'])
        report.append("")
    
    # 6. Scenario Analysis (if available)
    if scenario:
        report.append("6. SCENARIO ANALYSIS (WHAT-IF SIMULATION)")
        report.append("-" * 80)
        report.append(f"Current SEDI:         {scenario['current_sedi']:.2f}")
        report.append(f"Projected SEDI:       {scenario['estimated_sedi']:.2f}")
        report.append(f"Estimated Improvement: +{scenario['sedi_change']:.2f}")
        report.append(f"Current Category:     {scenario['current_category']}")
        report.append(f"Projected Category:   {scenario['estimated_category']}")
        report.append("")
        report.append("Applied Improvements:")
        for sector, pct in scenario['improvements_applied'].items():
            if pct > 0:
                report.append(f"  {sector.capitalize()}: +{pct}%")
        report.append("")
    
    report.append("="*80)
    report.append("END OF REPORT")
    report.append("="*80)
    
    return '\n'.join(report)


# ================================================================================
# STREAMLIT APP
# ================================================================================

def main():
    """Main Streamlit application."""
    
    # Header
    st.title("🗺️ Karnataka Deprivation Decision Support System")
    st.markdown("### Interactive Policy Analysis Platform")
    st.markdown("*District-Level Multidimensional Socioeconomic Deprivation Analysis*")
    st.markdown("---")
    
    # Initialize DSS
    @st.cache_resource
    def load_dss():
        try:
            BASE_DIR = Path(__file__).resolve().parent
            data_path = BASE_DIR / "data" / "module1_processed_data.csv"
            return DeprivationDSS(data_path)
            #return DeprivationDSS('output/module1_processed_data.csv')
        except FileNotFoundError:
            st.error("❌ Data file not found. Please ensure 'output/module1_processed_data.csv' exists.")
            st.stop()
    
    dss = load_dss()
    
    # Sidebar - District Selection and Controls
    st.sidebar.header("📍 District Selection")
    
    districts = dss.get_all_districts()
    selected_district = st.sidebar.selectbox(
        "Select District:",
        districts,
        index=0
    )
    
    st.sidebar.markdown("---")
    
    # State Overview Button
    if st.sidebar.button("🏛️ View State Overview"):
        st.session_state.view_mode = 'state'
    else:
        st.session_state.view_mode = 'district'
    
    st.sidebar.markdown("---")
    
    # Scenario Analysis Controls
    st.sidebar.header("🔧 Scenario Analysis")
    st.sidebar.markdown("*Simulate policy interventions (% improvement)*")
    
    education_improvement = st.sidebar.slider(
        "Education Improvement (%)",
        min_value=0,
        max_value=50,
        value=0,
        step=5,
        help="Increase in literacy rate and education scores"
    )
    
    health_improvement = st.sidebar.slider(
        "Health Improvement (%)",
        min_value=0,
        max_value=50,
        value=0,
        step=5,
        help="Increase in healthcare facilities"
    )
    
    infrastructure_improvement = st.sidebar.slider(
        "Infrastructure Improvement (%)",
        min_value=0,
        max_value=50,
        value=0,
        step=5,
        help="Increase in roads, electricity, urbanization"
    )
    
    economic_improvement = st.sidebar.slider(
        "Economic Improvement (%)",
        min_value=0,
        max_value=50,
        value=0,
        step=5,
        help="Increase in per capita income"
    )
    
    run_scenario = st.sidebar.button("▶️ Run Scenario Simulation", type="primary")
    
    # Main Content Area
    if st.session_state.get('view_mode') == 'state':
        # STATE OVERVIEW MODE
        st.header("🏛️ Karnataka State Overview")
        
        summary = dss.get_state_summary()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Districts", summary['total_districts'])
        with col2:
            st.metric("Average SEDI", f"{summary['avg_sedi']:.2f}")
        with col3:
            high_dep = summary['category_distribution'].get('High Deprivation', 0)
            st.metric("High Deprivation Districts", high_dep)
        
        st.markdown("---")
        
        # State visualizations
        st.subheader("📊 State-Level Distribution")
        fig_state = plot_state_overview(dss)
        st.pyplot(fig_state)
        
        st.markdown("---")
        
        # Top and bottom districts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🏆 Top 5 Districts (Lowest Deprivation)")
            top_5 = dss.data.nlargest(5, 'SEDI')[['District', 'SEDI', 'Deprivation_Category']]
            st.dataframe(top_5, use_container_width=True)
        
        with col2:
            st.subheader("⚠️ Bottom 5 Districts (Highest Deprivation)")
            bottom_5 = dss.data.nsmallest(5, 'SEDI')[['District', 'SEDI', 'Deprivation_Category']]
            st.dataframe(bottom_5, use_container_width=True)
        
    else:
        # DISTRICT-SPECIFIC MODE
        st.header(f"📍 District Analysis: {selected_district}")
        
        # Get district profile and recommendations
        profile = dss.get_district_profile(selected_district)
        deficits = dss.diagnose_sectoral_deficits(selected_district)
        recommendations = dss.generate_policy_recommendations(selected_district)
        
        # Display key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("SEDI Score", f"{profile['sedi']:.2f}/100")
        
        with col2:
            # Color-code category
            category_color = {
                'Low Deprivation': 'green',
                'Medium Deprivation': 'orange',
                'High Deprivation': 'red'
            }
            st.markdown(f"**Deprivation Category**")
            st.markdown(f":{category_color.get(profile['category'], 'blue')}[{profile['category']}]")
        
        with col3:
            st.metric("State Rank", f"{profile['rank']}/{profile['total_districts']}")
        
        with col4:
            # Color-code priority
            priority_color = {
                'URGENT': 'red',
                'HIGH': 'orange',
                'MODERATE': 'green'
            }
            st.markdown(f"**Priority Level**")
            st.markdown(f":{priority_color.get(recommendations['priority'], 'blue')}[{recommendations['priority']}]")
        
        st.markdown("---")
        
        # Tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Indicators", "💡 Recommendations", "🎯 Scenario Analysis", "📄 Report"])
        
        with tab1:
            st.subheader("Indicator Profile")
            
            # Show spatial cluster if available
            if profile['cluster_type'] != 'N/A':
                cluster_colors = {
                    'HH': 'green',
                    'LL': 'red',
                    'HL': 'blue',
                    'LH': 'orange'
                }
                cluster_labels = {
                    'HH': 'High-High (Low Deprivation Cluster)',
                    'LL': 'Low-Low (High Deprivation Cluster - HOTSPOT)',
                    'HL': 'High-Low (Positive Outlier)',
                    'LH': 'Low-High (Negative Outlier)'
                }
                st.info(f"**Spatial Cluster:** :{cluster_colors.get(profile['cluster_type'], 'blue')}[{profile['cluster_type']} - {cluster_labels.get(profile['cluster_type'], 'Unknown')}]")
            
            # Indicator chart
            fig_indicators = plot_district_indicators(profile)
            st.pyplot(fig_indicators)
            
            # Detailed indicator table
            st.markdown("#### Detailed Indicators")
            indicator_df = pd.DataFrame([
                {'Indicator': name, 'Value': f"{value:,.2f}"}
                for name, value in profile['indicators'].items()
            ])
            st.dataframe(indicator_df, use_container_width=True)
        
        with tab2:
            st.subheader("Policy Recommendations")
            
            # Priority alert
            if recommendations['priority'] == 'URGENT':
                st.error(f"🚨 **{recommendations['priority']} Priority** - Immediate intervention required")
            elif recommendations['priority'] == 'HIGH':
                st.warning(f"⚠️ **{recommendations['priority']} Priority** - Significant intervention needed")
            else:
                st.success(f"✅ **{recommendations['priority']} Priority** - Continued monitoring and support")
            
            # Sectoral recommendations
            st.markdown("#### Sector-Specific Interventions")
            
            if recommendations['sectors']:
                for sector, rec_info in recommendations['sectors'].items():
                    with st.expander(f"**{sector}** (Severity: {rec_info['severity']})"):
                        st.markdown(f"**Budget Priority:** {rec_info['budget_priority']}")
                        st.markdown("**Recommended Programs:**")
                        for i, program in enumerate(rec_info['programs'], 1):
                            st.markdown(f"{i}. {program}")
            else:
                st.info("No critical sectoral deficits identified. Continue monitoring and maintenance programs.")
            
            # Spatial recommendations
            if 'spatial_context' in recommendations:
                st.markdown("#### Spatial Context & Regional Coordination")
                st.info(recommendations['spatial_context'])
        
        with tab3:
            st.subheader("Scenario Analysis (What-If Simulation)")
            
            if run_scenario:
                improvements = {
                    'education': education_improvement,
                    'health': health_improvement,
                    'infrastructure': infrastructure_improvement,
                    'economic': economic_improvement
                }
                
                if sum(improvements.values()) == 0:
                    st.warning("⚠️ Please adjust at least one improvement slider to run a scenario.")
                else:
                    scenario = dss.simulate_scenario(selected_district, improvements)
                    
                    # Display scenario results
                    st.success("✅ Scenario simulation completed!")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Current SEDI", f"{scenario['current_sedi']:.2f}")
                    with col2:
                        st.metric("Projected SEDI", f"{scenario['estimated_sedi']:.2f}",
                                 delta=f"+{scenario['sedi_change']:.2f}")
                    with col3:
                        category_change = scenario['estimated_category'] != scenario['current_category']
                        if category_change:
                            st.metric("Category Change", scenario['estimated_category'],
                                     delta="Improved!" if scenario['estimated_sedi'] > scenario['current_sedi'] else "")
                        else:
                            st.metric("Category", scenario['current_category'], delta="No change")
                    
                    # Visualization
                    fig_scenario = plot_scenario_comparison(scenario)
                    st.pyplot(fig_scenario)
                    
                    # Store scenario in session state for report
                    st.session_state.scenario_results = scenario
            else:
                st.info("👈 Adjust improvement sliders in the sidebar and click 'Run Scenario Simulation'")
                st.markdown("""
                **How to use Scenario Analysis:**
                1. Select improvement percentages for each sector using sliders
                2. Click 'Run Scenario Simulation'
                3. Review projected SEDI score and category changes
                4. Use insights to prioritize policy interventions
                """)
        
        with tab4:
            st.subheader("District Report")
            
            # Generate report
            scenario_for_report = st.session_state.get('scenario_results', None)
            report_text = generate_district_report(dss, selected_district, recommendations, scenario_for_report)
            
            # Display report in text area
            st.text_area("Report Preview", report_text, height=400)
            
            # Download buttons
            col1, col2 = st.columns(2)
            
            with col1:
                # CSV download
                report_df = pd.DataFrame([profile['indicators']])
                report_df.insert(0, 'District', selected_district)
                report_df.insert(1, 'SEDI', profile['sedi'])
                report_df.insert(2, 'Category', profile['category'])
                
                csv = report_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Report (CSV)",
                    data=csv,
                    file_name=f"{selected_district}_report.csv",
                    mime="text/csv"
                )
            
            with col2:
                # Text report download
                st.download_button(
                    label="📥 Download Report (TXT)",
                    data=report_text,
                    file_name=f"{selected_district}_report.txt",
                    mime="text/plain"
                )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
    <small>Karnataka Deprivation Decision Support System | Rule-Based Policy Analysis | Version 1.0</small><br>
    <small>Data Source: Census 2011, NITI Aayog SDG Index 2021-22, Economic Survey Karnataka 2022-23</small>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()