
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="TellCo Telecom Analytics",
    page_icon="📊",
    layout="wide"
)

# =========================================================
# HELPER FUNCTIONS
# =========================================================

def format_number(value):
    if pd.isna(value):
        return "N/A"

    value = float(value)

    if abs(value) >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"
    elif abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    elif abs(value) >= 1_000:
        return f"{value / 1_000:.2f}K"
    else:
        return f"{value:.2f}"


def get_category_numeric_columns(data):
    numeric_cols = data.select_dtypes(include="number").columns.tolist()
    category_cols = [
        col for col in data.columns
        if col not in numeric_cols
    ]

    return category_cols, numeric_cols


# =========================================================
# LOAD DATA
# =========================================================

satisfaction = pd.read_csv("final_satisfaction_table.csv")
engagement = pd.read_csv("user_engagement.csv")
experience = pd.read_csv("user_experience.csv")
handsets = pd.read_csv("top_10_handsets.csv")
manufacturers = pd.read_csv("top_3_manufacturers.csv")
satisfied = pd.read_csv("top_10_satisfied.csv")

# MLflow tracking report if available
try:
    tracking = pd.read_csv("model_tracking_report.csv")
except:
    tracking = None


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("📌 Navigation")

page = st.sidebar.radio(
    "Select Dashboard",
    [
        "Overview",
        "User Engagement",
        "User Experience",
        "User Satisfaction",
        "Model Tracking"
    ]
)

st.sidebar.divider()
st.sidebar.caption("TellCo Telecom Customer Analytics")


# =========================================================
# OVERVIEW
# =========================================================

if page == "Overview":

    st.title("📊 TellCo Telecom Customer Analytics")
    st.subheader(
        "Customer Engagement, Experience & Satisfaction Dashboard"
    )

    st.divider()

    # KPI CARDS

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Customers",
        f"{len(satisfaction):,}"
    )

    col2.metric(
        "Top Handsets",
        "10"
    )

    col3.metric(
        "Top Manufacturers",
        "3"
    )

    col4.metric(
        "Avg Satisfaction",
        f"{satisfaction['Satisfaction_Score'].mean():.2f}"
    )

    st.divider()

    # -----------------------------------------------------
    # TOP HANDSETS
    # -----------------------------------------------------

    st.header("📱 Top 10 Handsets")

    category_cols, numeric_cols = get_category_numeric_columns(handsets)

    if category_cols and numeric_cols:

        handset_category = category_cols[0]
        handset_value = numeric_cols[0]

        fig = px.bar(
            handsets.sort_values(
                handset_value,
                ascending=True
            ),
            x=handset_value,
            y=handset_category,
            orientation="h",
            text=handset_value,
            title="Top 10 Handsets"
        )

        fig.update_traces(
            texttemplate="%{text}",
            textposition="outside"
        )

        fig.update_layout(
            xaxis_title="Number of Customers",
            yaxis_title="Handset",
            height=500
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # -----------------------------------------------------
    # TOP MANUFACTURERS
    # -----------------------------------------------------

    st.header("🏭 Top 3 Manufacturers")

    category_cols, numeric_cols = get_category_numeric_columns(
        manufacturers
    )

    if category_cols and numeric_cols:

        manufacturer_category = category_cols[0]
        manufacturer_value = numeric_cols[0]

        fig2 = px.bar(
            manufacturers.sort_values(
                manufacturer_value,
                ascending=True
            ),
            x=manufacturer_value,
            y=manufacturer_category,
            orientation="h",
            text=manufacturer_value,
            title="Top 3 Handset Manufacturers"
        )

        fig2.update_traces(
            texttemplate="%{text}",
            textposition="outside"
        )

        fig2.update_layout(
            xaxis_title="Number of Customers",
            yaxis_title="Manufacturer",
            height=350
        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )


# =========================================================
# USER ENGAGEMENT
# =========================================================

elif page == "User Engagement":

    st.title("📈 User Engagement Analysis")

    # -----------------------------------------------------
    # KPI CARDS
    # -----------------------------------------------------

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Avg Session Frequency",
        format_number(
            engagement["Sessions_Frequency"].mean()
        )
    )

    col2.metric(
        "Avg Total Duration",
        format_number(
            engagement["Total_Duration"].mean()
        )
    )

    col3.metric(
        "Avg Total Traffic",
        format_number(
            engagement["Total_Traffic"].mean()
        )
    )

    st.divider()

    # -----------------------------------------------------
    # KMEANS CLUSTERING
    # -----------------------------------------------------

    features = [
        "Sessions_Frequency",
        "Total_Duration",
        "Total_Traffic"
    ]

    engagement_model_data = engagement[features].copy()

    scaler = StandardScaler()

    scaled_data = scaler.fit_transform(
        engagement_model_data
    )

    kmeans = KMeans(
        n_clusters=3,
        random_state=42,
        n_init=10
    )

    engagement["Cluster"] = kmeans.fit_predict(
        scaled_data
    )

    cluster_counts = (
        engagement["Cluster"]
        .value_counts()
        .sort_index()
        .reset_index()
    )

    cluster_counts.columns = [
        "Cluster",
        "Customers"
    ]

    # Make sure all 3 clusters appear
    all_clusters = pd.DataFrame({
        "Cluster": [0, 1, 2]
    })

    cluster_counts = all_clusters.merge(
        cluster_counts,
        on="Cluster",
        how="left"
    )

    cluster_counts["Customers"] = (
        cluster_counts["Customers"]
        .fillna(0)
        .astype(int)
    )

    # -----------------------------------------------------
    # CLUSTER CHART
    # -----------------------------------------------------

    st.header("Engagement Cluster Distribution")

    fig = px.bar(
        cluster_counts,
        x="Cluster",
        y="Customers",
        text="Customers",
        title="Customers by Engagement Cluster"
    )

    fig.update_traces(
        textposition="outside"
    )

    fig.update_layout(
        xaxis_title="Cluster",
        yaxis_title="Customers"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # -----------------------------------------------------
    # CLUSTER TABLE
    # -----------------------------------------------------

    st.subheader("Engagement Cluster Summary")

    summary = (
        engagement
        .groupby("Cluster")[features]
        .mean()
        .reset_index()
    )

    st.dataframe(
        summary,
        use_container_width=True
    )


# =========================================================
# USER EXPERIENCE
# =========================================================

elif page == "User Experience":

    st.title("📶 User Experience Analysis")

    # -----------------------------------------------------
    # KPI CARDS
    # -----------------------------------------------------

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Avg TCP Retransmission",
        format_number(
            experience[
                "Avg_TCP_Retransmission"
            ].mean()
        )
    )

    col2.metric(
        "Avg RTT",
        format_number(
            experience["Avg_RTT"].mean()
        )
    )

    col3.metric(
        "Avg Throughput",
        format_number(
            experience["Avg_Throughput"].mean()
        )
    )

    st.divider()

    # -----------------------------------------------------
    # KMEANS
    # -----------------------------------------------------

    features = [
        "Avg_TCP_Retransmission",
        "Avg_RTT",
        "Avg_Throughput"
    ]

    experience_model_data = experience[
        features
    ].copy()

    scaler = StandardScaler()

    scaled_data = scaler.fit_transform(
        experience_model_data
    )

    kmeans = KMeans(
        n_clusters=3,
        random_state=42,
        n_init=10
    )

    experience["Cluster"] = kmeans.fit_predict(
        scaled_data
    )

    cluster_counts = (
        experience["Cluster"]
        .value_counts()
        .sort_index()
        .reset_index()
    )

    cluster_counts.columns = [
        "Cluster",
        "Customers"
    ]

    all_clusters = pd.DataFrame({
        "Cluster": [0, 1, 2]
    })

    cluster_counts = all_clusters.merge(
        cluster_counts,
        on="Cluster",
        how="left"
    )

    cluster_counts["Customers"] = (
        cluster_counts["Customers"]
        .fillna(0)
        .astype(int)
    )

    # -----------------------------------------------------
    # CHART
    # -----------------------------------------------------

    st.header("Experience Cluster Distribution")

    fig = px.bar(
        cluster_counts,
        x="Cluster",
        y="Customers",
        text="Customers",
        title="Customers by Experience Cluster"
    )

    fig.update_traces(
        textposition="outside"
    )

    fig.update_layout(
        xaxis_title="Cluster",
        yaxis_title="Customers"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # -----------------------------------------------------
    # EXPERIENCE SUMMARY
    # -----------------------------------------------------

    st.subheader("Experience Cluster Summary")

    summary = (
        experience
        .groupby("Cluster")[features]
        .mean()
        .reset_index()
    )

    st.dataframe(
        summary,
        use_container_width=True
    )


# =========================================================
# USER SATISFACTION
# =========================================================

elif page == "User Satisfaction":

    st.title("⭐ User Satisfaction Analysis")

    # -----------------------------------------------------
    # KPI CARDS
    # -----------------------------------------------------

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Average Engagement Score",
        f"{satisfaction['Engagement_Score'].mean():.2f}"
    )

    col2.metric(
        "Average Experience Score",
        f"{satisfaction['Experience_Score'].mean():.2f}"
    )

    col3.metric(
        "Average Satisfaction Score",
        f"{satisfaction['Satisfaction_Score'].mean():.2f}"
    )

    st.divider()

    # -----------------------------------------------------
    # SATISFACTION DISTRIBUTION
    # -----------------------------------------------------

    st.header("Satisfaction Score Distribution")

    upper_limit = satisfaction["Satisfaction_Score"].quantile(0.99)

    satisfaction_plot = satisfaction[
        satisfaction["Satisfaction_Score"] <= upper_limit
    ]

    fig = px.histogram(
        satisfaction_plot,
        x="Satisfaction_Score",
        nbins=40,
        title="Satisfaction Score Distribution (99th Percentile)"
    )

    fig.update_layout(
        xaxis_title="Satisfaction Score",
        yaxis_title="Number of Customers"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    outlier_count = len(satisfaction) - len(satisfaction_plot)

    st.info(
        f"{outlier_count:,} extreme observations are outside "
        f"the 99th percentile and are excluded from this visualization."
    )

    # -----------------------------------------------------
    # ENGAGEMENT VS EXPERIENCE
    # -----------------------------------------------------

    st.header("Engagement vs Experience")

    sample_data = satisfaction.sample(
        min(5000, len(satisfaction)),
        random_state=42
    )

    fig2 = px.scatter(
        sample_data,
        x="Engagement_Score",
        y="Experience_Score",
        color="Satisfaction_Score",
        title="Customer Engagement vs Experience"
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

    # -----------------------------------------------------
    # TOP 10
    # -----------------------------------------------------

    st.header("🏆 Top 10 Satisfied Customers")

    st.dataframe(
        satisfied,
        use_container_width=True
    )


# =========================================================
# MODEL TRACKING
# =========================================================

elif page == "Model Tracking":

    st.title("🤖 ML Model Tracking")

    # -----------------------------------------------------
    # LOAD REAL METRICS
    # -----------------------------------------------------

    if tracking is not None:

        latest = tracking.iloc[0]

        model_name = latest["Model"]
        mse_value = latest["MSE"]
        r2_value = latest["R2_Score"]
        code_version = latest["Code_Version"]
        source = latest["Source"]

    else:

        model_name = "Linear Regression"
        mse_value = 0
        r2_value = 1
        code_version = "v1.0"
        source = "Jupyter Notebook - Telecom Customer Analysis"

    # -----------------------------------------------------
    # MODEL KPI
    # -----------------------------------------------------

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Model",
        model_name
    )

    col2.metric(
        "R² Score",
        f"{r2_value:.2f}"
    )

    col3.metric(
        "MSE",
        f"{mse_value:.3e}"
    )

    st.divider()

    st.header("MLflow Tracking Information")

    tracking_info = pd.DataFrame({
        "Parameter": [
            "Model",
            "Features",
            "Code Version",
            "Source"
        ],
        "Value": [
            model_name,
            "Engagement Score, Experience Score",
            code_version,
            source
        ]
    })

    st.table(tracking_info)

    st.success(
        "Model tracking completed successfully using MLflow."
    )
