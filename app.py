"""Streamlit dashboard: Can Pune Breathe Better?   Run:  streamlit run app.py"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import ttest_ind
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from utils import load_data, daily_aqi, metrics, POLLUTANTS

st.set_page_config(page_title="Pune AQI Dashboard", layout="wide")
st.title("Can Pune Breathe Better?")
st.subheader("Pune Air Quality Analysis")


@st.cache_data
def get_data():
    return load_data()


@st.cache_data(show_spinner=False)
def prophet_forecast(daily, days):
    from prophet import Prophet

    def build():
        m = Prophet(daily_seasonality=False, weekly_seasonality=True,
                    yearly_seasonality=True, interval_width=0.95)
        try:
            m.add_country_holidays(country_name="IN")
        except Exception:
            pass
        return m

    n = int(len(daily) * 0.8)
    train, test = daily.iloc[:n], daily.iloc[n:]
    m = build(); m.fit(train)
    p = m.predict(m.make_future_dataframe(periods=len(test), freq="D"))["yhat"].iloc[-len(test):]
    res = metrics(test["y"].values, np.clip(p.values, 0, None))

    mf = build(); mf.fit(daily)
    fc = mf.predict(mf.make_future_dataframe(periods=days, freq="D"))
    for c in ["yhat", "yhat_lower", "yhat_upper"]:
        fc[c] = fc[c].clip(lower=0)
    return fc, res


df = get_data()
section = st.sidebar.radio("Navigate", [
    "Dataset Overview", "AQI Distribution", "Station-Wise Trends",
    "Seasonal Analysis", "Correlation", "Future Prediction"])

if section == "Dataset Overview":
    c1, c2, c3 = st.columns(3)
    c1.metric("Records", f"{len(df):,}")
    c2.metric("Average AQI", round(df["AQI"].mean(), 1))
    c3.metric("Max AQI", int(df["AQI"].max()))
    st.write("Dataset Overview")
    st.dataframe(df.head(20))
    st.write("Missing values per column")
    st.dataframe(df.isna().sum().rename("missing"))
    st.write("Summary statistics")
    st.dataframe(df.describe())

elif section == "AQI Distribution":
    fig = px.histogram(df, x="AQI", nbins=40, title="Air Quality Index Distribution")
    st.plotly_chart(fig, use_container_width=True)

elif section == "Station-Wise Trends":
    if "Location" not in df.columns:
        st.warning("No station/location column found in the dataset.")
    else:
        sdf = df.dropna(subset=["Location", "AQI"])
        avg = sdf.groupby("Location")["AQI"].mean().sort_values(ascending=False).reset_index()
        st.plotly_chart(px.bar(avg, x="Location", y="AQI", title="Average AQI by Station"),
                        use_container_width=True)
        station = st.selectbox("Select a station", sorted(sdf["Location"].unique()))
        s = sdf[sdf["Location"] == station].dropna(subset=["Date"])
        monthly = s.set_index("Date")["AQI"].resample("M").mean().reset_index()
        st.plotly_chart(px.line(monthly, x="Date", y="AQI", title=f"Monthly average AQI - {station}"),
                        use_container_width=True)

elif section == "Seasonal Analysis":
    d = daily_aqi(df)
    d["Month"] = d["Date"].dt.month
    monthly = d.groupby("Month")["AQI"].mean().reset_index()
    st.plotly_chart(px.bar(monthly, x="Month", y="AQI", title="Monthly Average AQI"),
                    use_container_width=True)
    winter = d[d["Month"].isin([12, 1, 2])]["AQI"]
    monsoon = d[d["Month"].isin([6, 7, 8, 9])]["AQI"]
    t, p = ttest_ind(winter, monsoon, equal_var=False)
    c1, c2, c3 = st.columns(3)
    c1.metric("Winter mean AQI", round(winter.mean(), 2))
    c2.metric("Monsoon mean AQI", round(monsoon.mean(), 2))
    c3.metric("Welch t-test p-value", f"{p:.4g}")
    st.caption("Winter = Dec-Feb, Monsoon = Jun-Sep. p < 0.05 means the difference is statistically significant.")

elif section == "Correlation":
    cols = [c for c in POLLUTANTS + ["AQI"] if c in df.columns]
    corr = df[cols].corr().round(2)
    fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                    title="Correlation Matrix of Pollutants and AQI")
    st.plotly_chart(fig, use_container_width=True)

elif section == "Future Prediction":
    st.title("Pune Air Quality Prediction")
    st.write("Estimates only - not official alerts or medical advice.")
    tab1, tab2 = st.tabs(["30-day forecast (Prophet)", "AQI from pollutants (Random Forest)"])

    with tab1:
        days = st.slider("Forecast horizon (days)", 7, 90, 30)
        d = daily_aqi(df).set_index("Date")["AQI"].asfreq("D").interpolate(limit_direction="both")
        daily = d.reset_index(); daily.columns = ["ds", "y"]
        try:
            with st.spinner("Training Prophet model..."):
                fc, res = prophet_forecast(daily, days)
            c1, c2, c3 = st.columns(3)
            c1.metric("R2 (test)", round(res["R2"], 3))
            c2.metric("MAE (test)", round(res["MAE"], 2))
            c3.metric("RMSE (test)", round(res["RMSE"], 2))
            fut = fc.iloc[-days:]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=daily["ds"], y=daily["y"], mode="markers",
                                     marker=dict(size=3, color="gray"), name="Actual AQI"))
            fig.add_trace(go.Scatter(x=fc["ds"], y=fc["yhat"], name="Model fit", line=dict(color="blue")))
            fig.add_trace(go.Scatter(x=fut["ds"], y=fut["yhat"], name="Forecast", line=dict(color="red", dash="dash")))
            fig.add_trace(go.Scatter(x=pd.concat([fc["ds"], fc["ds"][::-1]]),
                                     y=pd.concat([fc["yhat_upper"], fc["yhat_lower"][::-1]]),
                                     fill="toself", fillcolor="rgba(0,0,255,0.12)",
                                     line=dict(width=0), name="95% interval"))
            fig.update_layout(title="Prophet Forecast: Pune Daily Average AQI",
                              xaxis_title="Date", yaxis_title="AQI")
            st.plotly_chart(fig, use_container_width=True)
            st.write("Forecast table")
            st.dataframe(fut[["ds", "yhat", "yhat_lower", "yhat_upper"]].round(1)
                         .rename(columns={"ds": "Date", "yhat": "Predicted AQI",
                                          "yhat_lower": "Lower", "yhat_upper": "Upper"}))
            st.caption("Dataset ends in 2019, so this demonstrates the method; it is not a current prediction.")
        except ImportError:
            st.error("Prophet is not installed. Run: pip install prophet")

    with tab2:
        feats = [c for c in POLLUTANTS if c in df.columns]
        mdf = df.dropna(subset=feats + ["AQI"])
        X, y = mdf[feats], mdf["AQI"]
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
        with st.spinner("Training Random Forest..."):
            rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1).fit(Xtr, ytr)
        pred = rf.predict(Xte)
        r = metrics(yte, pred)
        c1, c2 = st.columns(2)
        c1.metric("Mean Absolute Error", round(r["MAE"], 2))
        c2.metric("R2", round(r["R2"], 3))
        res_df = pd.DataFrame({"Actual AQI": yte.values, "Predicted AQI": pred.round(2)})
        st.plotly_chart(px.scatter(res_df, x="Actual AQI", y="Predicted AQI",
                                   title="Actual vs Predicted AQI"), use_container_width=True)
        st.write("Try your own values")
        cols = st.columns(len(feats))
        vals = [cols[i].number_input(f, min_value=0.0, value=float(round(X[f].median(), 1)))
                for i, f in enumerate(feats)]
        st.success(f"Estimated AQI: {rf.predict(pd.DataFrame([vals], columns=feats))[0]:.0f}")

st.markdown("""
<style>
.footer {position: fixed; right: 20px; bottom: 10px; color: gray; font-size: 12px;
         text-align: right; background-color: white; padding: 5px;}
</style>
<div class="footer">Can Pune Breathe Better?<br>Pune Air Quality Analysis Dashboard</div>
""", unsafe_allow_html=True)