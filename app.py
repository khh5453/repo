# app.py
# -------------------------------------------------------------
# Google Analytics 세션 데이터 대시보드 (정리 버전, 시간대 필터 개선)
# -------------------------------------------------------------

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

st.set_page_config(page_title="사용자 행동 대시보드", layout="wide")

def wide_plot(fig, key=None, height=420):
    fig.update_layout(
        height=height,
        margin=dict(l=30, r=30, t=50, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    _, center, _ = st.columns([0.02, 0.96, 0.02])
    with center:
        st.plotly_chart(fig, use_container_width=True, key=key)

@st.cache_data
def load_data(path: str = "google.csv.gz"):
    df = pd.read_csv(path)
    df["visitStartTime"] = pd.to_datetime(df["visitStartTime"], errors="coerce")
    df["ym"]   = df["visitStartTime"].dt.to_period("M")
    df["date"] = df["visitStartTime"].dt.date
    df["hour"] = df["visitStartTime"].dt.hour

    not_campaign_tokens = {"(not set)", "not set", "(not provided)", "", "not available in demo dataset"}
    if "trafficCampaign" in df.columns:
        tc = df["trafficCampaign"].astype(str).str.strip().str.lower().fillna("")
        df["campaign_flag"] = np.where(~tc.isin(not_campaign_tokens), "캠페인 진행", "캠페인 미진행")
    else:
        # 컬럼이 없으면 기본값으로 안전하게 처리
        df["trafficCampaign"] = ""
        df["campaign_flag"] = "캠페인 미진행"

    for c in ["isFirstVisit", "isBounce", "addedToCart", "totalPageviews", "totalTimeOnSite"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    for c in ["country", "city", "trafficCampaign"]:
        if c in df.columns:
            df[c] = df[c].astype(str)
    return df

st.caption("App is up. Loading data…")  # 먼저 가벼운 프레임 출력

try:
    df = load_data()
except Exception as e:
    st.error("데이터 로딩 중 오류")
    st.exception(e)  # 원인 로그를 화면에 표시
    st.stop()

# 원천 기간 제한
start_bound = pd.Timestamp("2016-08-01")
end_bound   = pd.Timestamp("2017-06-01")
df = df[(df["visitStartTime"] >= start_bound) & (df["visitStartTime"] < end_bound)]

# -------------------- 사이드바 --------------------
if "sb_open" not in st.session_state:
    st.session_state.sb_open = True

# _, top_right = st.columns([0.7, 0.3])
# with top_right:
#     if st.button("🧰 필터 보이기/숨기기"):
#         st.session_state.sb_open = not st.session_state.sb_open

# 필터 마스크 초기화
mask = pd.Series(True, index=df.index)

if st.session_state.sb_open:
    with st.sidebar:
        st.header("📊 필터")

        # 1) 년월 범위 (카테고리 슬라이더)
        # all_months = pd.period_range(df["visitStartTime"].min(), df["visitStartTime"].max(), freq="M").astype(str).tolist()
        # m0, m1 = st.select_slider("년월 범위", options=all_months, value=(all_months[0], all_months[-1]))
        # p0, p1 = pd.Period(m0, "M"), pd.Period(m1, "M")
        # mask &= df["ym"].between(p0, p1)

        # 2) 일자 범위 (단일/구간 모두 지원)
        min_day, max_day = df.loc[mask, "date"].min(), df.loc[mask, "date"].max()
        day_sel = st.date_input("일자 (단일 또는 범위)", (min_day, max_day))
        if isinstance(day_sel, tuple):
            d0, d1 = [pd.to_datetime(d).date() for d in day_sel]
        else:
            d0 = d1 = pd.to_datetime(day_sel).date()
        mask &= df["date"].between(d0, d1)

        # 3) 시간대 필터 (0~23)
        h0, h1 = st.slider("시간대 (시)", 0, 23, (0, 23), step=1)
        mask &= df["hour"].between(h0, h1)

        # 4) 국가 → 도시 (계단식)
        # if "country" in df.columns:
        #     countries = sorted(df.loc[mask, "country"].dropna().unique().tolist())
        #     sel_countries = st.multiselect("국가", countries, default=countries)
        #     if sel_countries:
        #         mask &= df["country"].isin(sel_countries)

        # if "city" in df.columns:
        #     cities = sorted(df.loc[mask, "city"].dropna().unique().tolist())
        #     sel_cities = st.multiselect("도시", cities, default=cities)
        #     if sel_cities:
        #         mask &= df["city"].isin(sel_cities)

        # 5) 캠페인
        if "trafficCampaign" in df.columns:
            camp_all = sorted(df.loc[mask, "trafficCampaign"].dropna().unique().tolist())
            sel_camps = st.multiselect("캠페인 선택", camp_all, default=camp_all)
            if sel_camps:
                mask &= df["trafficCampaign"].isin(sel_camps)

# 최종 필터 적용
dff = df.loc[mask].copy()

# 단일 일자 선택 여부(그래프 축 단위 결정)
single_day = (d0 == d1)

# -------------------- KPI --------------------
st.title("📈 사용자 행동 대시보드")

total_users  = dff["fullVisitorId"].nunique()
bounce_rate  = dff["isBounce"].mean() if len(dff) else 0
new_rate     = (dff["isFirstVisit"] == 1).mean() if len(dff) else 0
revisit_rate = (dff["isFirstVisit"] == 0).mean() if len(dff) else 0
cart_conv    = (dff["addedToCart"] > 0).mean()   if len(dff) else 0

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Unique Users (Carrying Capacity)", f"{total_users:,}")
k2.metric("Bounce Rate (세션 단위)", f"{bounce_rate:.1%}")
k3.metric("신규 유입 비율 (세션 단위)", f"{new_rate:.1%}")
k4.metric("Cart 전환율 (세션 단위)", f"{cart_conv:.1%}")
k5.metric("재방문율 (세션 단위)", f"{revisit_rate:.1%}")

# ============== 1) Carrying Capacity ==============
with st.container(border=True):
    st.subheader("1. 기간별 Carrying Capacity")
    if single_day:
        cap = dff.groupby("hour")["fullVisitorId"].nunique().reset_index(name="unique_users")
        xcol, xlabel = "hour", "시"
    else:
        cap = dff.groupby("ym")["fullVisitorId"].nunique().reset_index(name="unique_users")
        cap["ym_str"] = cap["ym"].astype(str)
        xcol, xlabel = "ym_str", "년-월"
    fig1 = px.line(cap, x=xcol, y="unique_users", markers=True)
    fig1.update_layout(xaxis_title=xlabel, yaxis_title="Unique Users (명)")
    wide_plot(fig1, key="cap", height=430)

# ============== 2) Bounce + Retain (dual) ==============
with st.container(border=True):
    st.subheader("2. 기간별 Bounce Rate + Retain Rate (이중축)")
    if single_day:
        br = dff.groupby("hour")["isBounce"].mean().reset_index().rename(columns={"isBounce":"bounce_rate"})
        br["retain_rate"] = 1 - br["bounce_rate"]; xcol="hour"; xlabel="시"
    else:
        br = dff.groupby("ym")["isBounce"].mean().reset_index().rename(columns={"isBounce":"bounce_rate"})
        br["retain_rate"] = 1 - br["bounce_rate"]; br["ym_str"]=br["ym"].astype(str); xcol="ym_str"; xlabel="년-월"
    fig2 = make_subplots(specs=[[{"secondary_y": True}]])
    fig2.add_trace(go.Scatter(x=br[xcol], y=br["bounce_rate"], mode="lines+markers", name="Bounce Rate"), secondary_y=False)
    fig2.add_trace(go.Scatter(x=br[xcol], y=br["retain_rate"], mode="lines+markers", name="Retain Rate"), secondary_y=True)
    fig2.update_xaxes(title_text=xlabel)
    fig2.update_yaxes(title_text="Bounce Rate", tickformat=".0%", secondary_y=False)
    fig2.update_yaxes(title_text="Retain Rate", tickformat=".0%", secondary_y=True)
    wide_plot(fig2, key="bounce_retain", height=430)

# ============== 3) 캠페인별 재방문율 ==============
with st.container(border=True):
    st.subheader("3. 캠페인 진행여부별 재방문율 (세션 단위)")
    rev = (dff.assign(revisit=(dff["isFirstVisit"] == 0).astype(int))
              .groupby("campaign_flag")["revisit"].mean().reset_index())
    fig3 = px.bar(rev, x="campaign_flag", y="revisit", text_auto=".1%")
    fig3.update_layout(xaxis_title="캠페인 진행 여부", yaxis_title="재방문율 (세션 단위)")
    fig3.update_yaxes(tickformat=".0%")
    wide_plot(fig3, key="revisit_campaign", height=420)

# ============== 4) Cart Conversion (dual) ==============
with st.container(border=True):
    st.subheader("4. 기간별 Cart 전환율 + 캠페인 진행에 따른 전환율 (이중축)")
    dff_cart = dff.assign(cart=(dff["addedToCart"] > 0).astype(int))
    if single_day:
        all_line  = dff_cart.groupby("hour")["cart"].mean().reset_index(); all_line_x="hour"; xlabel="시"
        by_flag   = dff_cart.groupby(["hour","campaign_flag"])["cart"].mean().reset_index(); split_key="hour"
    else:
        all_line  = dff_cart.groupby("ym")["cart"].mean().reset_index(); all_line["ym_str"]=all_line["ym"].astype(str); all_line_x="ym_str"; xlabel="년-월"
        by_flag   = dff_cart.groupby(["ym","campaign_flag"])["cart"].mean().reset_index(); by_flag["ym_str"]=by_flag["ym"].astype(str); split_key="ym_str"
    fig4 = make_subplots(specs=[[{"secondary_y": True}]])
    fig4.add_trace(go.Scatter(x=all_line[all_line_x], y=all_line["cart"], mode="lines+markers", name="전체 전환율"), secondary_y=False)
    for cf, sub in by_flag.groupby("campaign_flag"):
        fig4.add_trace(go.Scatter(x=sub[split_key], y=sub["cart"], mode="lines+markers", name=f"{cf} 전환율", line=dict(dash="dash")), secondary_y=True)
    fig4.update_xaxes(title_text=xlabel)
    fig4.update_yaxes(title_text="전체 전환율", tickformat=".0%", secondary_y=False)
    fig4.update_yaxes(title_text="캠페인별 전환율", tickformat=".0%", secondary_y=True)
    wide_plot(fig4, key="cart_dual", height=440)

# ============== 5) Stickiness (pageviews mean) ==============
with st.container(border=True):
    st.subheader("5. 고착도 (평균 페이지뷰 기준, 세션 단위)")
    if single_day:
        stick = dff.groupby("hour")["totalPageviews"].mean().reset_index(); xcol="hour"; xlabel="시"
    else:
        stick = dff.groupby("ym")["totalPageviews"].mean().reset_index(); stick["ym_str"]=stick["ym"].astype(str); xcol="ym_str"; xlabel="년-월"
    fig5 = px.line(stick, x=xcol, y="totalPageviews", markers=True,
                   labels={xcol: xlabel, "totalPageviews": "평균 페이지뷰/세션"})
    wide_plot(fig5, key="stickiness", height=420)


# ============================================================
# 6) 기간별 카트 전환율 (Dual): trafficSource Top5 vs deviceCategory
#    - 좌축: trafficSource Top5 라인
#    - 우축: deviceCategory 라인(Desktop/Mobile/Tablet 등)
#    - 단일 일자 선택 시 '시간대별(시)', 다중 일자/기간이면 '월별'
# ============================================================
with st.container(border=True):
    st.subheader("6. 기간별 카트 전환율 — Source Top5 (좌) vs Device (우)")

    if dff.empty:
        st.info("표시할 데이터가 없습니다.")
    else:
        # 시간 축 결정 (단일 일자면 시간대, 아니면 월)
        single_day_auto = dff["visitStartTime"].dt.normalize().nunique() == 1
        dff = dff.assign(hour=dff["visitStartTime"].dt.hour,
                         ym=dff["visitStartTime"].dt.to_period("M"))
        time_key = "hour" if single_day_auto else "ym"
        x_label  = "시" if single_day_auto else "년-월"

        # 카트 전환 플래그
        dff_cart = dff.assign(cart=(dff["addedToCart"] > 0).astype(int))

        # trafficSource Top5 (세션 수 기준)
        top5_sources = (dff_cart.groupby("trafficSource")["cart"]
                        .size().sort_values(ascending=False).head(5).index.tolist())
        ts = (dff_cart[dff_cart["trafficSource"].isin(top5_sources)]
              .groupby([time_key, "trafficSource"])["cart"]
              .mean().reset_index())
        if time_key == "ym":
            ts["x"] = ts["ym"].astype(str)
        else:
            ts["x"] = ts["hour"]

        # deviceCategory
        dc = (dff_cart.groupby([time_key, "deviceCategory"])["cart"]
              .mean().reset_index())
        if time_key == "ym":
            dc["x"] = dc["ym"].astype(str)
        else:
            dc["x"] = dc["hour"]

        # ---- Dual Axis Figure ----
        from plotly.subplots import make_subplots
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # 좌축: Source Top5
        for src, sub in ts.groupby("trafficSource"):
            fig.add_trace(
                go.Scatter(x=sub["x"], y=sub["cart"],
                           mode="lines+markers", name=f"SRC: {src}"),
                secondary_y=False
            )

        # 우축: Device
        for dev, sub in dc.groupby("deviceCategory"):
            fig.add_trace(
                go.Scatter(x=sub["x"], y=sub["cart"],
                           mode="lines+markers", name=f"DEV: {dev}",
                           line=dict(dash="dash")),
                secondary_y=True
            )

        fig.update_xaxes(title_text=x_label)
        fig.update_yaxes(title_text="카트 전환율 (Source Top5)", tickformat=".0%", secondary_y=False)
        fig.update_yaxes(title_text="카트 전환율 (Device)",      tickformat=".0%", secondary_y=True)
        wide_plot(fig, key="dual_cart_src_dev", height=460)



# ============================================================
# 7) 기간별 고착도 — Source Top5 (좌) vs Device (우)  [Dual Axis]
#    - 좌축: trafficSource Top5의 평균 페이지뷰/세션
#    - 우축: deviceCategory의 평균 페이지뷰/세션
#    - 단일 일자 선택 시 '시간대별(시)', 그 외에는 '월별'
# ============================================================
with st.container(border=True):
    st.subheader("7. 기간별 고착도 — Source Top5 (좌) vs Device (우)")

    if dff.empty:
        st.info("표시할 데이터가 없습니다.")
    else:
        df_stick = dff.copy()
        df_stick["hour"] = df_stick["visitStartTime"].dt.hour
        df_stick["ym"]   = df_stick["visitStartTime"].dt.to_period("M")

        single_day_auto = df_stick["visitStartTime"].dt.normalize().nunique() == 1
        time_key = "hour" if single_day_auto else "ym"
        x_label  = "시" if single_day_auto else "년-월"

        # ---- 좌축: trafficSource Top5 (세션 수 기준) ----
        top5_sources = (df_stick.groupby("trafficSource")
                        .size().sort_values(ascending=False).head(5).index.tolist())

        src = (df_stick[df_stick["trafficSource"].isin(top5_sources)]
               .groupby([time_key, "trafficSource"])["totalPageviews"]
               .mean().reset_index(name="pv"))

        if time_key == "ym":
            src["x"] = src["ym"].astype(str)
        else:
            src["x"] = src["hour"]

        # ---- 우축: deviceCategory ----
        dev = (df_stick.groupby([time_key, "deviceCategory"])["totalPageviews"]
               .mean().reset_index(name="pv"))
        if time_key == "ym":
            dev["x"] = dev["ym"].astype(str)
        else:
            dev["x"] = dev["hour"]

        # ---- Dual Axis Figure ----
        from plotly.subplots import make_subplots
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # 좌축: Source Top5 라인
        for src_name, sub in src.groupby("trafficSource"):
            fig.add_trace(
                go.Scatter(x=sub["x"], y=sub["pv"],
                           mode="lines+markers", name=f"SRC: {src_name}"),
                secondary_y=False
            )

        # 우축: Device 라인(점선)
        for dev_name, sub in dev.groupby("deviceCategory"):
            fig.add_trace(
                go.Scatter(x=sub["x"], y=sub["pv"],
                           mode="lines+markers", name=f"DEV: {dev_name}",
                           line=dict(dash="dash")),
                secondary_y=True
            )

        fig.update_xaxes(title_text=x_label)
        fig.update_yaxes(title_text="평균 페이지뷰/세션 (Source Top5)", secondary_y=False)
        fig.update_yaxes(title_text="평균 페이지뷰/세션 (Device)",      secondary_y=True)
        wide_plot(fig, key="dual_stickiness_src_dev", height=460)





