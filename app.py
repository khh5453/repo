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

st.set_page_config(page_title="Google Merchandise 활성 사용자 분석", layout="wide")

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
def load_data(path: str = "final.csv.gz"):
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

# 필터 마스크 초기화
mask = pd.Series(True, index=df.index)

if st.session_state.sb_open:
    with st.sidebar:
        st.header("📊 필터")

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
        

# 최종 필터 적용
dff = df.loc[mask].copy()

# 단일 일자 선택 여부(그래프 축 단위 결정)
single_day = (d0 == d1)

# -------------------- KPI --------------------
st.title("📈 Google Merchandise 활성 사용자 분석")

dff['date'] = dff['visitStartTime'].dt.to_period("d")
DAU = dff.groupby("date")["fullVisitorId"].nunique()
dff["month"] = dff["visitStartTime"].dt.to_period("M")
MAU = dff.groupby("month")["fullVisitorId"].nunique()

stickiness = (DAU.mean() / MAU.mean())

avg_session_duration = dff["totalTimeOnSite"].mean()

added_users = dff[dff["addedToCart"] == 1].groupby("date")["fullVisitorId"].nunique()

cart_conversion_rate_dau = (added_users / DAU).fillna(0)
cart_conversion_rate = cart_conversion_rate_dau.mean()

# 카드 여백/텍스트 살짝 손질
st.markdown("""
<style>
.kpi-card { padding: 10px 12px 4px 12px; }
.kpi-card [data-testid="stMetric"] { margin: 0; }
.kpi-card [data-testid="stMetricValue"]{ font-size: 1.6rem; }
.kpi-card [data-testid="stMetricDelta"]{ font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)

c1, c2, c3, c4, c5 = st.columns(5, gap="large")

with c1:
    with st.container(border=True):
        st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
        st.metric("DAU (평균)", f"{int(round(DAU.mean())):,}")
        st.caption("일별 고유 사용자 (평균)")
        st.markdown('</div>', unsafe_allow_html=True)

with c2:
    with st.container(border=True):
        st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
        st.metric("MAU (평균)", f"{int(round(MAU.mean())):,}")
        st.caption("월별 고유 사용자 (평균)")
        st.markdown('</div>', unsafe_allow_html=True)

with c3:
    with st.container(border=True):
        st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
        st.metric("고착도", f"{stickiness:.1%}")
        st.caption("DAU / MAU (평균)")
        st.markdown('</div>', unsafe_allow_html=True)

with c4:
    with st.container(border=True):
        st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
        st.metric("평균 체류시간 (초)", f"{int(round(avg_session_duration)):,}")
        st.caption("체류시간 (평균)")
        st.markdown('</div>', unsafe_allow_html=True)

with c5:
    with st.container(border=True):
        st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
        st.metric("일간 카트 전환율", f"{cart_conversion_rate:.1%}")
        st.caption("장바구니 담은 유저수 / DAU (평균)")
        st.markdown('</div>', unsafe_allow_html=True)



# ==========================================
# (2열 배치) ⬅️ 왼쪽: 활성 사용자 & 고착도  |  오른쪽: 체류시간 / 카트 전환율 (라디오 토글)
#  - 왼쪽 토글: 월간(MAU) / 일간(DAU) / 고착도
#  - 오른쪽 토글: 평균 체류시간(월별) ↔ 카트 전환율(월별)
# ==========================================
with st.container(border=True):
    # st.subheader("활성 사용자 · 고착도  vs  체류시간 / 카트 전환율")

    colL, colR = st.columns(2, gap="large")

    # ---------- 공통 데이터 준비 ----------
    base_all = dff.copy() if "dff" in globals() and isinstance(dff, pd.DataFrame) else df.copy()
    need_common = {"visitStartTime", "fullVisitorId"}
    if base_all.empty or not need_common.issubset(base_all.columns):
        st.info("표시할 데이터가 없습니다. (visitStartTime, fullVisitorId 필요)")
    else:
        base_all["visitStartTime"] = pd.to_datetime(base_all["visitStartTime"], errors="coerce")
        base_all = base_all.dropna(subset=["visitStartTime"])
        base_all["fullVisitorId"] = base_all["fullVisitorId"].astype(str)
        base_all["day"]   = base_all["visitStartTime"].dt.floor("D")
        base_all["month"] = base_all["visitStartTime"].dt.to_period("M").dt.to_timestamp(how="start")

        # ===== 왼쪽: MAU/DAU/고착도 =====
        with colL:
            st.markdown("### 기간별 활성 사용자 & 고착도")

            # MAU (월간 고유 사용자)
            mau = (base_all.groupby("month")["fullVisitorId"]
                          .nunique()
                          .reset_index(name="MAU")
                          .sort_values("month"))
            if not mau.empty:
                months_full = pd.date_range(mau["month"].min(), mau["month"].max(), freq="MS")
                mau = (mau.set_index("month").reindex(months_full, fill_value=0)
                         .rename_axis("month").reset_index())
                mau["month_str"] = mau["month"].dt.strftime("%Y-%m")

            # DAU (일간 고유 사용자)
            dau = (base_all.groupby("day")["fullVisitorId"]
                           .nunique()
                           .reset_index(name="DAU")
                           .sort_values("day"))
            if not dau.empty:
                days_full = pd.date_range(dau["day"].min(), dau["day"].max(), freq="D")
                dau = (dau.set_index("day").reindex(days_full, fill_value=0)
                         .rename_axis("day").reset_index())
                dau["day_str"] = dau["day"].dt.strftime("%Y-%m-%d")

            # 고착도 = (해당 월의 '일별 DAU 평균') / MAU
            if not dau.empty and not mau.empty:
                dau_avg_m = (dau.assign(month=dau["day"].dt.to_period("M").dt.to_timestamp(how="start"))
                                .groupby("month")["DAU"].mean().reset_index(name="DAU_avg"))
                stick = (mau.merge(dau_avg_m, on="month", how="left")
                            .assign(stickiness=lambda x: np.where(x["MAU"]>0, x["DAU_avg"]/x["MAU"], np.nan)))
                stick["month_str"] = stick["month"].dt.strftime("%Y-%m")
            else:
                stick = pd.DataFrame(columns=["month", "month_str", "stickiness"])

            viewL = st.radio("보기", ("월간(MAU)", "일간(DAU)", "고착도"),
                             index=0, horizontal=True, key="left_mau_dau_stick")

            if viewL == "월간(MAU)":
                plot_df, xcol, ycol = mau, "month_str", "MAU"
                ytitle = "MAU (월간 고유 사용자)"
            elif viewL == "일간(DAU)":
                plot_df, xcol, ycol = dau, "day_str", "DAU"
                ytitle = "DAU (일간 고유 사용자)"
            else:
                plot_df, xcol, ycol = stick, "month_str", "stickiness"
                ytitle = "고착도 (월평균 DAU / MAU)"

            if plot_df.empty:
                st.info("선택한 보기 모드에 데이터가 없습니다.")
            else:
                figL = px.line(plot_df, x=xcol, y=ycol, markers=True,
                               labels={xcol: "기간", ycol: ytitle})
                if viewL == "고착도":
                    figL.update_yaxes(tickformat=".0%")
                    figL.update_traces(hovertemplate="%{x}<br>"+ytitle+": %{y:.1%}<extra></extra>")
                else:
                    figL.update_yaxes(rangemode="tozero", separatethousands=True)
                    figL.update_traces(hovertemplate="%{x}<br>"+ytitle+": %{y:,}<extra></extra>")
                figL.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10), showlegend=False)
                st.plotly_chart(figL, use_container_width=True, key="plot_left_mau_dau_stick")

        # ===== 오른쪽: 평균 체류시간(월) / 카트 전환율(월) 토글 =====
        with colR:
            st.markdown("### 기간별 체류시간 & 카트 전환율")

            need_right = {"totalTimeOnSite", "addedToCart"}
            if not need_right.issubset(base_all.columns):
                st.info("totalTimeOnSite 또는 addedToCart 컬럼이 없어 표시할 수 없습니다.")
            else:
                base_all["totalTimeOnSite"] = pd.to_numeric(base_all["totalTimeOnSite"], errors="coerce")
                base_all["addedToCart"]     = pd.to_numeric(base_all["addedToCart"], errors="coerce").fillna(0)

                viewR = st.radio("보기", ("평균 체류시간(월별)", "카트 전환율(월별)"),
                                 index=0, horizontal=True, key="right_dwell_cart_toggle")

                if viewR == "평균 체류시간(월별)":
                    # 월별 평균 체류시간(초 → 분)
                    grp = (base_all.groupby("month")["totalTimeOnSite"]
                                   .mean().reset_index(name="avg_sec")
                                   .sort_values("month"))
                    if grp.empty:
                        st.info("데이터가 없습니다.")
                    else:
                        months_full = pd.date_range(grp["month"].min(), grp["month"].max(), freq="MS")
                        grp = (grp.set_index("month").reindex(months_full)
                                 .rename_axis("month").reset_index())
                        grp["avg_min"] = grp["avg_sec"] / 60.0
                        grp["x"] = grp["month"].dt.strftime("%Y-%m")

                        figR = px.line(grp, x="x", y="avg_min", markers=True,
                                       custom_data=["avg_sec"],
                                       labels={"x": "월", "avg_min": "평균 체류시간(분)"})
                        figR.update_traces(
                            hovertemplate="%{x}<br>평균 체류시간: %{y:.1f}분 (%{customdata[0]:.0f}초)<extra></extra>"
                        )
                        figR.update_yaxes(rangemode="tozero")

                else:  # "카트 전환율(월별)"
                    # 월별 활성 사용자(MAU)
                    mau = base_all.groupby("month")["fullVisitorId"].nunique()
                    # 월별 장바구니 추가 사용자(고유)
                    add_m = base_all.loc[base_all["addedToCart"] > 0] \
                                     .groupby("month")["fullVisitorId"].nunique()

                    if mau.empty:
                        st.info("데이터가 없습니다.")
                    else:
                        months_full = pd.date_range(mau.index.min(), mau.index.max(), freq="MS")
                        mau = mau.reindex(months_full, fill_value=0)
                        add_m = add_m.reindex(months_full, fill_value=0)

                        rate = np.where(mau > 0, add_m / mau, 0.0)
                        out = pd.DataFrame({
                            "period": months_full,
                            "rate": rate,
                            "added_users": add_m.values,
                            "active_users": mau.values
                        })
                        out["x"] = out["period"].dt.strftime("%Y-%m")

                        figR = px.line(
                            out, x="x", y="rate", markers=True,
                            labels={"x": "월", "rate": "카트 전환율(월간)"}
                        )
                        figR.update_yaxes(tickformat=".0%", rangemode="tozero")
                        figR.update_traces(
                            hovertemplate=(
                                "%{x}<br>"
                                "카트 전환율: %{y:.1%}<br>"
                                "장바구니 추가 사용자: %{customdata[0]:,}명<br>"
                                "활성 사용자: %{customdata[1]:,}명<extra></extra>"
                            ),
                            customdata=out[["added_users", "active_users"]].to_numpy()
                        )

                if 'figR' in locals():
                    figR.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10), showlegend=False)
                    st.plotly_chart(figR, use_container_width=True, key="plot_right_dwell_or_cart")






# ==========================================
# (2열 배치) ⬅️ 왼쪽: 국가별 MAU/DAU/고착도  |  오른쪽: 국가별 체류시간/카트전환율 (월별)
#  - 공통: Top10 국가 중 멀티선택 (한 번만 선택해서 양쪽 그래프에 적용)
#  - 왼쪽 토글: MAU(월) / DAU(일) / 고착도(월)
#  - 오른쪽 토글: 평균 체류시간(분, 월별) / 카트 전환율(월별, 고유 사용자 기준)
# ==========================================
with st.container(border=True):

    base = dff.copy() if "dff" in globals() and isinstance(dff, pd.DataFrame) else df.copy()
    need = {"visitStartTime", "fullVisitorId", "country"}
    if base.empty or not need.issubset(base.columns):
        st.info("표시할 데이터가 없습니다. (visitStartTime, fullVisitorId, country 필요)")
    else:
        # ---- 공통 전처리 ----
        base["visitStartTime"] = pd.to_datetime(base["visitStartTime"], errors="coerce")
        base = base.dropna(subset=["visitStartTime", "country"])
        base["fullVisitorId"] = base["fullVisitorId"].astype(str)
        base["country"] = base["country"].astype(str)
        base["month"] = base["visitStartTime"].dt.to_period("M").dt.to_timestamp(how="start")
        base["day"]   = base["visitStartTime"].dt.floor("D")

        # Top10 국가 (전체 기간 고유 사용자 기준)
        top10 = (
            base.groupby("country")["fullVisitorId"]
                .nunique()
                .sort_values(ascending=False)
                .head(10)
                .index.tolist()
        )

        if not top10:
            st.info("Top10 국가가 없습니다.")
        else:
            # 하나만 보여 공통 적용
            sel_countries = st.multiselect(
                "국가 선택 (Top10)", options=top10, default=top10[:3], key="country_top10_shared"
            )
            countries = sel_countries if sel_countries else top10
            sub = base[base["country"].isin(countries)].copy()

            if sub.empty:
                st.info("선택한 국가에 데이터가 없습니다.")
            else:
                # 누락 기간 보정용 축
                months = pd.date_range(sub["month"].min(), sub["month"].max(), freq="MS")
                days   = pd.date_range(sub["day"].min(),   sub["day"].max(),   freq="D")
                mi_month_country = pd.MultiIndex.from_product([months, countries], names=["month", "country"])
                mi_day_country   = pd.MultiIndex.from_product([days,   countries], names=["day",   "country"])

                colL, colR = st.columns(2, gap="large")

                # ========== ⬅️ 왼쪽: MAU / DAU / 고착도 ==========
                with colL:
                    st.markdown("### 국가별 활성 사용자 & 고착도")

                    # MAU
                    mau = (
                        sub.groupby(["month", "country"])["fullVisitorId"]
                           .nunique().reset_index(name="mau")
                           .set_index(["month", "country"])
                           .reindex(mi_month_country, fill_value=0)
                           .reset_index()
                    )
                    mau["month_str"] = mau["month"].dt.strftime("%Y-%m")

                    # DAU
                    dau = (
                        sub.groupby(["day", "country"])["fullVisitorId"]
                           .nunique().reset_index(name="dau")
                           .set_index(["day", "country"])
                           .reindex(mi_day_country, fill_value=0)
                           .reset_index()
                    )
                    dau["day_str"] = dau["day"].dt.strftime("%Y-%m-%d")

                    # 고착도 = (월평균 DAU) / MAU
                    dau_avg_m = (
                        dau.assign(month=dau["day"].dt.to_period("M").dt.to_timestamp(how="start"))
                           .groupby(["month", "country"])["dau"]
                           .mean().reset_index(name="dau_avg")
                           .set_index(["month", "country"])
                           .reindex(mi_month_country, fill_value=0)
                           .reset_index()
                    )
                    stick = mau.merge(dau_avg_m, on=["month", "country"], how="left")
                    stick["stickiness"] = np.where(stick["mau"] > 0, stick["dau_avg"] / stick["mau"], np.nan)
                    stick["month_str"] = stick["month"].dt.strftime("%Y-%m")

                    viewL = st.radio("보기", ("MAU(월별)", "DAU(일별)", "고착도(월별)"),
                                     index=0, horizontal=True, key="country_left_view")

                    if viewL == "MAU(월별)":
                        plot_df, xcol, ycol = mau, "month_str", "mau"
                        ytitle, yfmt_pct = "MAU (월간 고유 사용자)", False
                    elif viewL == "DAU(일별)":
                        plot_df, xcol, ycol = dau, "day_str", "dau"
                        ytitle, yfmt_pct = "DAU (일별 고유 사용자)", False
                    else:
                        plot_df, xcol, ycol = stick, "month_str", "stickiness"
                        ytitle, yfmt_pct = "고착도 (월평균 DAU / MAU)", True

                    figL = px.line(
                        plot_df, x=xcol, y=ycol,
                        color="country", markers=True,
                        labels={xcol: "기간", ycol: ytitle, "country": "국가"}
                    )
                    figL.update_layout(
                        height=420, margin=dict(l=10, r=10, t=30, b=10),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    if yfmt_pct:
                        figL.update_yaxes(tickformat=".0%")
                        figL.update_traces(hovertemplate="%{x}<br>%{fullData.name}<br>"+ytitle+": %{y:.1%}<extra></extra>")
                    else:
                        figL.update_yaxes(rangemode="tozero", separatethousands=True)
                        figL.update_traces(hovertemplate="%{x}<br>%{fullData.name}<br>"+ytitle+": %{y:,}<extra></extra>")

                    try:
                        wide_plot(figL, key="country_left_mau_dau_stick", height=420)
                    except NameError:
                        st.plotly_chart(figL, use_container_width=True, key="country_left_mau_dau_stick")

                # ========== ➡️ 오른쪽: 체류시간 / 카트 전환율 ==========
                with colR:
                    st.markdown("### 국가별 체류시간 & 카트 전환율")

                    need_right = {"totalTimeOnSite", "addedToCart"}
                    if not need_right.issubset(sub.columns):
                        st.info("totalTimeOnSite 또는 addedToCart 컬럼이 없어 표시할 수 없습니다.")
                    else:
                        sub["totalTimeOnSite"] = pd.to_numeric(sub["totalTimeOnSite"], errors="coerce")
                        sub["addedToCart"]     = pd.to_numeric(sub["addedToCart"], errors="coerce").fillna(0)

                        viewR = st.radio("보기", ("평균 체류시간(분)", "카트 전환율"),
                                         index=0, horizontal=True, key="country_right_view")

                        # 평균 체류시간(분)
                        dwell = (
                            sub.groupby(["month", "country"])["totalTimeOnSite"]
                               .mean().reset_index(name="avg_sec")
                               .set_index(["month", "country"])
                               .reindex(mi_month_country)
                               .reset_index()
                        )
                        dwell["value"] = dwell["avg_sec"] / 60.0
                        dwell["month_str"] = dwell["month"].dt.strftime("%Y-%m")

                        # 카트 전환율(고유 사용자 기준)
                        active = (
                            sub.groupby(["month", "country"])["fullVisitorId"]
                               .nunique().reset_index(name="active_users")
                               .set_index(["month", "country"])
                               .reindex(mi_month_country, fill_value=0)
                        )
                        added  = (
                            sub[sub["addedToCart"] > 0]
                               .groupby(["month", "country"])["fullVisitorId"]
                               .nunique().reset_index(name="added_users")
                               .set_index(["month", "country"])
                               .reindex(mi_month_country, fill_value=0)
                        )
                        conv = (active.join(added, how="left").fillna(0).reset_index())
                        conv["rate"] = np.where(conv["active_users"] > 0,
                                                conv["added_users"] / conv["active_users"], 0.0)
                        conv["month_str"] = conv["month"].dt.strftime("%Y-%m")

                        if viewR == "평균 체류시간(분)":
                            plot_df = dwell.copy()
                            ycol, ytitle, yfmt_pct = "value", "평균 체류시간(분)", False
                        else:
                            plot_df = conv.copy()
                            ycol, ytitle, yfmt_pct = "rate", "카트 전환율", True

                        figR = px.line(
                            plot_df, x="month_str", y=ycol,
                            color="country", markers=True,
                            labels={"month_str": "월", ycol: ytitle, "country": "국가"}
                        )
                        figR.update_layout(
                            height=420, margin=dict(l=10, r=10, t=30, b=10),
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                        )
                        if yfmt_pct:
                            figR.update_yaxes(tickformat=".0%")
                            figR.update_traces(
                                hovertemplate=("%{x}<br>%{fullData.name}<br>"
                                               + ytitle + ": %{y:.1%}<extra></extra>")
                            )
                        else:
                            figR.update_yaxes(rangemode="tozero")
                            figR.update_traces(
                                hovertemplate=("%{x}<br>%{fullData.name}<br>"
                                               + ytitle + ": %{y:.1f}분<extra></extra>")
                            )

                        try:
                            wide_plot(figR, key="country_right_dwell_cart", height=420)
                        except NameError:
                            st.plotly_chart(figR, use_container_width=True, key="country_right_dwell_cart")




# ==========================================
# (2열 배치) ⬅️ 왼쪽: 6개 국가 × Traffic Medium 개수  |  오른쪽: Traffic Medium 추이(이중축)
#  - 왼쪽 토글: 그룹막대 / 누적막대
#  - 오른쪽 토글: 월별 / 일별
# ==========================================
from plotly.subplots import make_subplots

with st.container(border=True):

    base = dff.copy() if "dff" in globals() and isinstance(dff, pd.DataFrame) else df.copy()
    need_left  = {"country", "trafficMedium"}
    need_right = {"visitStartTime", "trafficMedium"}

    if base.empty or not (need_left | need_right).issubset(base.columns):
        st.info("표시할 데이터가 없습니다. (country, trafficMedium, visitStartTime 필요)")
    else:
        # 공통 전처리
        base["country"]       = base["country"].astype(str)
        base["trafficMedium"] = base["trafficMedium"].astype(str)

        # ✅ none / (not set) / not set / (none) / cpc / cpm 제거 (대소문자/공백 무시)
        drop_set = {"none", "(not set)", "not set", "(none)", "cpc", "cpm"}
        tm_norm = base["trafficMedium"].str.strip().str.lower()
        base = base[~tm_norm.isin(drop_set)].copy()

        colL, colR = st.columns(2, gap="large")

        # ========== ⬅️ 왼쪽: 6개 국가 × Traffic Medium 개수 ==========
        with colL:
            st.markdown("### 국가별 유입 경로")

            # 🇰🇷→🇺🇸 매핑(데이터셋 표기)
            country_map = {
                "미국": "United States",
                "인도": "India",
                "베트남": "Vietnam",
                "태국": "Thailand",
                "터키": "Turkey",
                "브라질": "Brazil",
            }
            countries_order = [country_map[k] for k in ["미국", "인도", "베트남", "태국", "터키", "브라질"]]

            # 실제 존재하는 6개만 사용
            countries_present = [c for c in countries_order if c in base["country"].unique().tolist()]
            sub_left = base[base["country"].isin(countries_present)].copy()

            if sub_left.empty:
                st.info("선택한 6개 국가 중 데이터가 없습니다.")
            else:
                # 공백 medium 정리
                sub_left["trafficMedium"] = sub_left["trafficMedium"].str.strip().replace({"": "(unknown)"})

                mediums = sorted(sub_left["trafficMedium"].unique().tolist())
                mi = pd.MultiIndex.from_product([countries_present, mediums], names=["country", "trafficMedium"])
                counts = (
                    sub_left.groupby(["country", "trafficMedium"])
                            .size()
                            .reindex(mi, fill_value=0)
                            .reset_index(name="count")
                )

                # ✅ 국가 정렬: 총 count 합계 기준 내림차순
                country_order = (counts.groupby("country")["count"]
                                 .sum()
                                 .sort_values(ascending=False)
                                 .index.tolist())

                # (옵션) ✅ 범례(trafficMedium)도 합계 기준 내림차순으로
                medium_order = (counts.groupby("trafficMedium")["count"]
                                .sum()
                                .sort_values(ascending=False)
                                .index.tolist())

                # ✅ 카테고리 순서 고정
                counts["country"] = pd.Categorical(counts["country"],
                                                   categories=country_order,
                                                   ordered=True)

                mode = st.radio("표시 방식", ("그룹막대", "누적막대"),
                                index=0, horizontal=True, key="bar_mode_medium_l")

                # ✅ x축/범례 순서를 category_orders로 확정
                fig_left = px.bar(
                    counts,  # 이미 카테고리 순서 고정 → 추가 sort 불필요
                    x="country", y="count", color="trafficMedium",
                    text="count",
                    labels={"country": "Country", "count": "Count", "trafficMedium": "Traffic Medium"},
                    category_orders={
                        "country": country_order,
                        "trafficMedium": medium_order,  # 필요 없으면 이 줄만 주석 처리
                    },
                )
                fig_left.update_traces(texttemplate="%{text:,}", textposition="outside", cliponaxis=False)
                fig_left.update_layout(
                    barmode="group" if mode == "그룹막대" else "relative",
                    height=420,
                    margin=dict(l=10, r=10, t=30, b=10),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                )
                fig_left.update_yaxes(rangemode="tozero", separatethousands=True)
                fig_left.update_traces(hovertemplate="%{x}<br>%{fullData.name}<br>Count: %{y:,}<extra></extra>")
                st.plotly_chart(fig_left, use_container_width=True, key="country6_medium_counts_col")



        # ========== ➡️ 오른쪽: Traffic Medium 추이(이중축) ==========
        with colR:
            st.markdown("### 유입 경로별 추이")

            sub_right = base.copy()
            # 시간 컬럼
            sub_right["visitStartTime"] = pd.to_datetime(sub_right["visitStartTime"], errors="coerce")
            sub_right = sub_right.dropna(subset=["visitStartTime"])
            # medium 표준화
            sub_right["medium_norm"] = sub_right["trafficMedium"].str.strip().str.lower()

            target_set = {"referral", "organic", "affiliate"}
            sub_right = sub_right[sub_right["medium_norm"].isin(target_set)]

            if sub_right.empty:
                st.info("'referral' / 'organic' / 'affiliate' 데이터가 없습니다.")
            else:
                sub_right["day"]   = sub_right["visitStartTime"].dt.floor("D")
                sub_right["month"] = sub_right["visitStartTime"].dt.to_period("M").dt.to_timestamp(how="start")

                view = st.radio("보기", ("월별", "일별"), index=0, horizontal=True, key="tm_view_dual_r")
                if view == "월별":
                    time_col = "month"; xfmt = "%Y-%m"; freq = "MS"
                else:
                    time_col = "day";   xfmt = "%Y-%m-%d"; freq = "D"

                agg = (
                    sub_right.groupby([time_col, "medium_norm"])
                             .size().reset_index(name="sessions")
                )
                all_times = pd.date_range(agg[time_col].min(), agg[time_col].max(), freq=freq)

                mediums = ["referral", "organic", "affiliate"]
                mi = pd.MultiIndex.from_product([all_times, mediums], names=[time_col, "medium_norm"])
                agg_full = (
                    agg.set_index([time_col, "medium_norm"])
                       .reindex(mi, fill_value=0)
                       .reset_index()
                       .rename(columns={time_col: "t"})
                )
                agg_full["x"] = agg_full["t"].dt.strftime(xfmt)

                fig_right = make_subplots(specs=[[{"secondary_y": True}]])
                # 좌축: referral, organic
                for name, dash in [("referral", None), ("organic", "dot")]:
                    line_df = agg_full[agg_full["medium_norm"] == name]
                    if not line_df.empty:
                        fig_right.add_trace(
                            go.Scatter(
                                x=line_df["x"], y=line_df["sessions"],
                                mode="lines+markers",
                                name=name.capitalize(),
                                line=dict(dash=dash) if dash else None
                            ),
                            secondary_y=False
                        )
                # 우축: affiliate
                aff_df = agg_full[agg_full["medium_norm"] == "affiliate"]
                if not aff_df.empty:
                    fig_right.add_trace(
                        go.Scatter(
                            x=aff_df["x"], y=aff_df["sessions"],
                            mode="lines+markers",
                            name="Affiliate",
                            line=dict(width=2)
                        ),
                        secondary_y=True
                    )

                fig_right.update_xaxes(title_text="기간")
                fig_right.update_yaxes(title_text="Sessions (referral / organic)", secondary_y=False)
                fig_right.update_yaxes(title_text="Sessions (affiliate)", secondary_y=True)
                fig_right.update_traces(hovertemplate="%{x}<br>%{fullData.name}: %{y:,}<extra></extra>")
                fig_right.update_layout(
                    height=420,
                    margin=dict(l=10, r=10, t=30, b=10),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig_right, use_container_width=True, key="tm_ref_org_aff_dual_col")





# ==========================================
# (2열 배치) ⬅️ 왼쪽: trafficSource Top3 비중(월별)  |  ➡️ 오른쪽: 국가별 세션수 — Direct·YouTube·Google
# ==========================================
with st.container(border=True):

    if "dff" in globals() and isinstance(dff, pd.DataFrame):
        base = dff.copy()
    else:
        base = df.copy()

    # 공통 안전 체크
    need_left  = {"trafficSource", "visitStartTime"}
    need_right = {"country", "trafficSource"}
    if base.empty or not (need_left | need_right).issubset(base.columns):
        st.info("표시할 데이터가 없습니다. (visitStartTime, trafficSource, country 필요)")
    else:
        # 공통 전처리
        base["visitStartTime"] = pd.to_datetime(base["visitStartTime"], errors="coerce")
        base = base.dropna(subset=["visitStartTime"])
        base["trafficSource"]  = base["trafficSource"].astype(str).fillna("(unknown)").replace({"": "(unknown)"})
        base["country"]        = base["country"].astype(str)

        colL, colR = st.columns(2, gap="large")

        # -----------------------------
        # ⬅️ 왼쪽: trafficSource Top3 비중 (월별)
        # -----------------------------
        with colL:
            st.markdown("### 기간별 Traffic Source Top3")

            df_src = base.copy()
            df_src["ym"] = df_src["visitStartTime"].dt.to_period("M")

            # Top3 산출 (전체 기간 합계 상위 3개)
            top3 = (
                df_src["trafficSource"]
                .value_counts(dropna=False)
                .head(3)
                .index
                .tolist()
            )

            if not top3:
                st.info("Top3 소스를 찾지 못했습니다.")
            else:
                # 월별 소스별 세션 수
                monthly_src = (
                    df_src[df_src["trafficSource"].isin(top3)]
                    .groupby(["ym", "trafficSource"])
                    .size()
                    .rename("cnt")
                    .reset_index()
                )

                if monthly_src.empty:
                    st.info("선택된 기간/필터에 해당하는 소스 데이터가 없습니다.")
                else:
                    # 월별 전체 세션 수
                    monthly_total = df_src.groupby("ym").size().rename("total").reset_index()

                    # 비중 계산 = cnt / total
                    merged = monthly_src.merge(monthly_total, on="ym", how="left")
                    merged["share"] = np.where(merged["total"] > 0, merged["cnt"] / merged["total"], 0.0)

                    # 누락 월 0 채우기 → 라인 끊김 방지
                    all_months = pd.period_range(df_src["ym"].min(), df_src["ym"].max(), freq="M")
                    piv = (
                        merged.pivot(index="ym", columns="trafficSource", values="share")
                              .reindex(all_months)
                              .fillna(0.0)
                    )
                    piv["ym_str"] = piv.index.astype(str)
                    long = (
                        piv.reset_index(drop=True)
                           .melt(id_vars="ym_str", var_name="trafficSource", value_name="share")
                    )

                    figL = px.line(
                        long,
                        x="ym_str",
                        y="share",
                        color="trafficSource",
                        markers=True,
                        labels={"ym_str": "년-월", "share": "비중", "trafficSource": "Traffic Source (Top3)"},
                    )
                    figL.update_yaxes(tickformat=".0%", rangemode="tozero")
                    figL.update_traces(hovertemplate="%{x}<br>%{legendgroup}: %{y:.1%}<extra></extra>")
                    figL.update_layout(
                        height=420,
                        margin=dict(l=10, r=10, t=30, b=10),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        xaxis=dict(categoryorder="array", categoryarray=sorted(long["ym_str"].unique())),
                    )
                    st.plotly_chart(figL, use_container_width=True, key="ts_top3_share_monthly_col")

        # -----------------------------
        # ➡️ 오른쪽: 국가별 세션수 — Direct · YouTube · Google (정규화 매칭)
        # -----------------------------
        with colR:
            st.markdown("### 국가별 Traffic Source Top3")

            # 지정 6개 국가 (데이터에 존재하는 것만 사용)
            country_map = {
                "미국": "United States",
                "인도": "India",
                "베트남": "Vietnam",
                "태국": "Thailand",
                "터키": "Turkey",
                "브라질": "Brazil",
            }
            countries_order = [country_map[k] for k in ["미국", "인도", "베트남", "태국", "터키", "브라질"]]
            present = [c for c in countries_order if c in base["country"].unique().tolist()]

            sub = base[base["country"].isin(present)].copy()
            if sub.empty:
                st.info("선택한 6개 국가 중 데이터가 없습니다.")
            else:
                # trafficSource 정규화(부분매칭)
                src_lower = sub["trafficSource"].str.lower().str.strip()
                sub["src3"] = "(other)"
                sub.loc[src_lower.str.contains("direct",  na=False), "src3"] = "Direct"
                sub.loc[src_lower.str.contains("youtube", na=False), "src3"] = "YouTube"
                sub.loc[src_lower.str.contains("google",  na=False), "src3"] = "Google"
                sub = sub[sub["src3"].isin(["Direct", "YouTube", "Google"])]

                if sub.empty:
                    st.info("Direct / YouTube / Google 데이터가 없습니다.")
                else:
                    # 모든 (국가 × 소스) 조합 0 채우기
                    mi = pd.MultiIndex.from_product([present, ["Direct", "YouTube", "Google"]],
                                                    names=["country", "src3"])
                    counts = (
                        sub.groupby(["country", "src3"])
                        .size()
                        .reindex(mi, fill_value=0)
                        .reset_index(name="count")
                    )
                    counts["country"] = pd.Categorical(counts["country"], categories=present, ordered=True)

                    figR = px.bar(
                        counts.sort_values(["country", "count"], ascending=[True, False]),
                        x="country", y="count", color="src3", text="count",
                        category_orders={"country": present, "src3": ["Direct", "YouTube", "Google"]},
                        labels={"country": "Country", "count": "Sessions", "src3": "Traffic Source"},
                        # 🔴🟦 색상 지정: Direct(미지정=기존색 유지), YouTube=빨간색, Google=하늘색
                        color_discrete_map={"Direct" : "#046aca", "YouTube": "red", "Google": "skyblue"},
                    )

                    figR.update_traces(texttemplate="%{text:,}", textposition="outside", cliponaxis=False)
                    figR.update_yaxes(rangemode="tozero", separatethousands=True)
                    figR.update_layout(
                        barmode="group",
                        height=420,
                        margin=dict(l=10, r=10, t=30, b=10),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    )

                    st.plotly_chart(figR, use_container_width=True, key="country6_src3_counts_norm_col")



