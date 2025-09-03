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
def load_data(path: str = "google.csv"):
    df = pd.read_csv(path)
    df["visitStartTime"] = pd.to_datetime(df["visitStartTime"], errors="coerce")
    df["ym"]   = df["visitStartTime"].dt.to_period("M")
    df["date"] = df["visitStartTime"].dt.date
    df["hour"] = df["visitStartTime"].dt.hour

    not_campaign_tokens = {"(not set)", "not set", "(not provided)", "", "not available in demo dataset"}
    tc = df["trafficCampaign"].astype(str).str.strip().str.lower().fillna("")
    df["campaign_flag"] = np.where(~tc.isin(not_campaign_tokens), "캠페인 진행", "캠페인 미진행")

    for c in ["isFirstVisit", "isBounce", "addedToCart", "totalPageviews", "totalTimeOnSite"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    for c in ["country", "city", "trafficCampaign"]:
        if c in df.columns:
            df[c] = df[c].astype(str)
    return df

df = load_data()

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
# 6) 지역별 지표 (일자별): 전환율(카트) 또는 재방문율
#     - 본문(사이드바 아님)에 필터 UI 배치
#     - 지역 레벨: country / city
#     - 단일 국가 선택 후 도시 다중선택 가능
#     - 표시할 지역 수 Top-N 옵션, 7일 이동평균 스위치
# ============================================================
with st.container(border=True):
    st.subheader("6. 지역별 지표 (일자별)")

    data_geo = dff.copy()
    data_geo["date"] = pd.to_datetime(data_geo["visitStartTime"]).dt.date

    # --- 상단 인라인 필터 (본문) ---
    f1, f2, f3, f4 = st.columns([1.2, 1.1, 1.1, 1.0])
    with f1:
        metric_opt = st.selectbox("지표", ["카트 전환율", "재방문율"], index=0, key="geo_metric")
    with f2:
        level = st.radio("지역 레벨", ["country", "city"], index=0, horizontal=True, key="geo_level")
    with f3:
        topn = st.slider("표시 지역 수 (Top-N)", 3, 20, 8, 1, help="세션 수 기준 상위 N개 지역만 표시")
    with f4:
        smooth = st.toggle("7일 이동평균", value=False, key="geo_smooth")

    # --- 지역 선택 위젯 (본문, 계단식: country → city) ---
    sel_countries, sel_cities = None, None

    if level == "country":
        # 나라 단위 분석: 국가 다중선택
        country_pool = sorted(data_geo["country"].dropna().unique().tolist())
        sel_countries = st.multiselect(
            "국가 선택",
            country_pool,
            # 이전 선택 유지 (있으면), 없으면 앞에서 8개만 기본 선택
            default=st.session_state.get("geo_countries", country_pool[:min(8, len(country_pool))]),
            key="geo_countries",
        )
        if sel_countries:
            data_geo = data_geo[data_geo["country"].isin(sel_countries)]
        dim = "country"

    else:  # level == "city"
        # 1) 먼저 '국가'를 복수 선택 → 2) 해당 국가의 '도시'만 후보로 제시
        ccol, tcol = st.columns([1.2, 2.0])
        with ccol:
            country_pool = sorted(data_geo["country"].dropna().unique().tolist())
            sel_countries = st.multiselect(
                "국가 선택(도시 후보 제한)",
                country_pool,
                default=st.session_state.get("geo_city_countries", []),
                key="geo_city_countries",
            )

        # 국가 선택 결과로 city 후보를 제한
        if sel_countries:
            city_source = data_geo[data_geo["country"].isin(sel_countries)]
        else:
            city_source = data_geo

        city_pool = sorted(city_source["city"].dropna().unique().tolist())

        # 이전에 고른 도시들 중 여전히 후보에 있는 것만 기본값으로 유지
        prev_cities = st.session_state.get("geo_cities", [])
        default_cities = [c for c in prev_cities if c in city_pool] or city_pool[:min(10, len(city_pool))]

        with tcol:
            sel_cities = st.multiselect(
                "도시 선택",
                city_pool,
                default=default_cities,
                key="geo_cities",
            )

        # 실제 데이터 제한
        if sel_countries:
            data_geo = data_geo[data_geo["country"].isin(sel_countries)]
        if sel_cities:
            data_geo = data_geo[data_geo["city"].isin(sel_cities)]

        dim = "city"


    # --- 지표 계산 ---
    if metric_opt == "카트 전환율":
        data_geo["metric"] = (data_geo["addedToCart"] > 0).astype(int)
        y_title = "카트 전환율"
    else:  # 재방문율
        data_geo["metric"] = (data_geo["isFirstVisit"] == 0).astype(int)
        y_title = "재방문율"

    # --- 일자별 집계 (지역 x 날짜)
    grp = data_geo.groupby(["date", dim]).agg(
        value=("metric", "mean"),
        sessions=("fullVisitorId", "size")
    ).reset_index()

    # 선택 안 했을 때는 세션 수 기준 Top-N만 표시 (라인 난립 방지)
    if level == "country" and not sel_countries:
        top_dims = grp.groupby(dim)["sessions"].sum().sort_values(ascending=False).head(topn).index
        grp = grp[grp[dim].isin(top_dims)]
    if level == "city" and not sel_cities:
        top_dims = grp.groupby(dim)["sessions"].sum().sort_values(ascending=False).head(topn).index
        grp = grp[grp[dim].isin(top_dims)]

    # --- 7일 이동평균 옵션 ---
    grp = grp.sort_values(["date", dim])
    grp["date"] = pd.to_datetime(grp["date"])
    y_col = "value"
    if smooth:
        grp["ma7"] = grp.groupby(dim)["value"].transform(lambda s: s.rolling(7, min_periods=1).mean())
        y_col = "ma7"

    # --- 라인 차트 ---
    label_dim = "국가" if level == "country" else "도시"
    fig_geo = px.line(
        grp, x="date", y=y_col, color=dim, markers=True,
        labels={"date": "일자", y_col: y_title, dim: label_dim}
    )
    fig_geo.update_yaxes(tickformat=".0%")
    wide_plot(fig_geo, key="geo_metric_plot", height=460)
