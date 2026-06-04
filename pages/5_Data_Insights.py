import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os

st.set_page_config(page_title="Data Insights", page_icon="📈", layout="wide")

st.title("📈 Data Insights")
st.markdown("Exploratory analysis of the CMS Medicare dataset and synthetic claims data.")

# Load synthetic data
@st.cache_data
def load_synthetic():
    base = os.path.dirname(os.path.dirname(__file__))
    return pd.read_csv(os.path.join(base, 'data/synthetic/synthetic_claims.csv'))

@st.cache_data
def load_cms():
    base = os.path.dirname(os.path.dirname(__file__))
    return pd.read_csv(
        os.path.join(base, 'data/raw/PHY_R26_P05_V10_D24_Prov_Svc.csv'),
        nrows=500000,
        low_memory=False
    )

df_synthetic = load_synthetic()
df_cms = load_cms()

st.markdown("---")

# Chart 1 — Denial rate by specialty
st.markdown("### Chart 1 — Claim Denial Rate by Provider Specialty")
st.caption("Which specialties have the highest denial rates?")

denial_by_specialty = (df_synthetic.groupby('provider_specialty')['is_denied']
                         .mean()
                         .sort_values(ascending=True)
                         .reset_index())
denial_by_specialty.columns = ['Specialty', 'Denial Rate']
overall_avg = df_synthetic['is_denied'].mean()

fig1 = px.bar(denial_by_specialty, x='Denial Rate', y='Specialty',
              orientation='h',
              title='Claim Denial Rate by Provider Specialty')
fig1.add_vline(x=overall_avg, line_dash='dash', line_color='red',
               annotation_text='Overall Average')
st.plotly_chart(fig1, use_container_width=True)
st.caption("Psychiatry, Pain Management, and Oncology show denial rates 8-10% above the overall average.")

st.markdown("---")

# Chart 2 — Filing delay vs denial rate
st.markdown("### Chart 2 — Filing Delay vs Denial Rate")
st.caption("Does late filing predict denial?")

bins = [0, 7, 14, 30, 60, 90]
labels = ['0-7', '8-14', '15-30', '31-60', '61-90']
df_synthetic['filing_bucket'] = pd.cut(
    df_synthetic['days_to_submission'],
    bins=bins, labels=labels, include_lowest=True)

denial_by_delay = (df_synthetic.groupby('filing_bucket')['is_denied']
                     .mean().reset_index())
denial_by_delay.columns = ['Days to Submission', 'Denial Rate']

fig2 = px.line(denial_by_delay, x='Days to Submission', y='Denial Rate',
               markers=True, title='Claim Denial Rate by Days to Submission')
fig2.add_hline(y=overall_avg, line_dash='dash', line_color='red',
               annotation_text='Overall Average')
st.plotly_chart(fig2, use_container_width=True)
st.caption("Claims filed after 60 days show nearly double the denial rate of on-time claims.")

st.markdown("---")

# Chart 3 — Top 20 HCPCS codes
st.markdown("### Chart 3 — Top 20 HCPCS Codes by Total Services")
st.caption("What procedures are most common in Medicare billing?")

top20_cpt = (df_cms.groupby(['HCPCS_Cd', 'HCPCS_Desc'])['Tot_Srvcs']
               .sum()
               .sort_values(ascending=False)
               .head(20)
               .reset_index())
top20_cpt['HCPCS_Short'] = (top20_cpt['HCPCS_Cd'] + ' - ' +
                              top20_cpt['HCPCS_Desc'].str[:40])

fig3 = px.bar(top20_cpt, x='Tot_Srvcs', y='HCPCS_Short',
              orientation='h',
              title='Top 20 HCPCS Codes by Total Services (millions)')
fig3.update_layout(yaxis={'categoryorder': 'total ascending'})
st.plotly_chart(fig3, use_container_width=True)
st.caption("Factor VIII dominates due to per-unit billing. Office visit E&M codes represent the most common patient encounters.")

st.markdown("---")

# Summary stats
st.markdown("### Dataset Summary")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Synthetic Claims", "50,000")
col2.metric("Overall Denial Rate", f"{df_synthetic['is_denied'].mean():.1%}")
col3.metric("CMS Rows Sampled", "500,000")
col4.metric("Unique Specialties", df_synthetic['provider_specialty'].nunique())