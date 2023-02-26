import os

import altair as alt
import numpy as np
import pandas as pd
import requests
import streamlit as st

from opal.conf.conf import MODEL_DIR
from opal.score.collaborative_filtering import NeuMF

st.set_page_config(page_title="Opal | o!m AI Score Predictor", page_icon=":comet:")


@st.cache_resource
def get_model(model_path=MODEL_DIR / "V2_2023_01/checkpoints/epoch=5-step=43584.ckpt"):
    net = NeuMF.load_from_checkpoint(model_path.as_posix())
    net.eval()
    return net


def random_uid():
    st.session_state['uid'] = np.random.choice(net.uid_le.classes_).split("/")[0]


def random_mid():
    st.session_state['mid'] = np.random.choice(net.mid_le.classes_).split("/")[0]


def predict(uid, mid):
    try:
        return float(net.predict(uid, mid))
    except ValueError:
        return None


@st.cache_data
def get_username(user_id: int):
    url = f'https://osu.ppy.sh/api/get_user?k={api_key}&u={user_id}'

    response = requests.get(url)

    if response.status_code == 200:
        user_data = response.json()[0]
        username = user_data['username']
        return username
    else:
        return None


@st.cache_data
def get_beatmap_metadata(beatmap_id: int):
    url = f'https://osu.ppy.sh/api/get_beatmaps?k={api_key}&b={beatmap_id}'

    response = requests.get(url)

    if response.status_code == 200:
        user_data = response.json()[0]
        map_metadata = f"{user_data['artist']} - " \
                       f"{user_data['title']} (" \
                       f"{user_data['creator']}) [" \
                       f"{user_data['version']}]"
        return map_metadata
    else:
        return None


MODEL_YEAR = 2023
YEARS_SEARCH = 10
DEFAULT_USER_ID = "2193881"
DEFAULT_MAP_ID = "767046"
net = get_model()
uid = DEFAULT_USER_ID
mid = DEFAULT_MAP_ID
api_key = os.environ['OSU_API']

st.markdown("""
<h1 style='text-align: center;'>
<span style='filter: drop-shadow(0 0.2mm 1mm rgba(142, 190, 255, 0.9));'>Opal</span>

<p style='color:grey'>AI Score Predictor by 
<a href='https://github.com/Eve-ning/' style='text-decoration:none'>Evening</a>

<a href='https://github.com/Eve-ning/opal' style='text-decoration:none'>![Repo](https://img.shields.io/badge/GitHub-opal-success?logo=github)</a>
<a href='https://twitter.com/dev_evening' style='text-decoration:none'>![Twitter](https://img.shields.io/badge/-dev__evening-blue?logo=twitter)</a>
<a href='https://github.com/Eve-ning/opal/blob/master/models/' style='text-decoration:none'>
![Model Size](https://img.shields.io/github/size/Eve-ning/opal/models/V2_2023_01/checkpoints/epoch%253D5-step%253D43584.ckpt?color=purple&label=Model%20Size&logo=pytorch-lightning)
</a>
</p>
</h1>
""", unsafe_allow_html=True)
left, right = st.columns(2)
with left:
    st.button("Get Random Player", on_click=random_uid)
    uid = st.text_input("User ID", key='uid', placeholder="2193881")
with right:
    st.button("Get Random Map", on_click=random_mid)
    mid = st.text_input("Map ID", key='mid', placeholder="767046")

try:
    uid = int(uid)
    mid = int(mid)
except:
    st.stop()
username = get_username(uid)
map_metadata = get_beatmap_metadata(mid)

preds = []
years = range(MODEL_YEAR - YEARS_SEARCH, MODEL_YEAR)
speeds = {-1: 'HT', 0: 'NT', 1: 'DT'}
for speed, speed_txt in speeds.items():
    for year in years:
        uid_ = f"{uid}/{year}"
        mid_ = f"{mid}/{speed}"
        if uid_ in net.uid_le.classes_:
            pred = predict(uid_, mid_)
            if pred:
                preds.append([year, speed_txt, pred])

df_pred = pd.DataFrame(preds, columns=['year', 'speed', 'pred'])

df_pred_last = df_pred[df_pred['year'] == df_pred['year'].max()]
if not df_pred_last.empty:
    year_last = list(df_pred_last['year'])[0]
else:
    year_last = "??"
st.markdown(f"""
<h5 style='text-align: center;'>
If <a href='https://osu.ppy.sh/users/{uid}' style='text-decoration:none'>{username}</a> 
played 
<a href='https://osu.ppy.sh/b/{mid}' style='text-decoration:none'>{map_metadata}</a> in {year_last}
</h5>
""", unsafe_allow_html=True)

pred_ht = df_pred_last.loc[(df_pred_last['speed'] == 'HT'), 'pred']
pred_nt = df_pred_last.loc[(df_pred_last['speed'] == 'NT'), 'pred']
pred_dt = df_pred_last.loc[(df_pred_last['speed'] == 'DT'), 'pred']

pred_ht = float(pred_ht) if pred_ht.any() else None
pred_nt = float(pred_nt) if pred_nt.any() else None
pred_dt = float(pred_dt) if pred_dt.any() else None

c1, c2, c3 = st.columns(3)

st.markdown("---")

if pred_ht:
    delta_ht = pred_ht - pred_nt
    c1.metric(f"{':warning:' if delta_ht < 0 else ''} "
              f"HT", f"{pred_ht:.2%}",
              delta=f"{delta_ht:.2%}")
else:
    c1.warning(":warning: No Prediction")

if pred_nt:
    c2.metric(f"NT", f"{pred_nt:.2%}")
else:
    c2.warning(":warning: No Prediction")

if pred_dt:
    delta_dt = pred_dt - pred_nt
    c3.metric(f"{':warning:' if delta_dt > 0 else ''} "
              f"DT", f"{pred_dt:.2%}", delta=f"{delta_dt:.2%}")
else:
    c3.warning(":warning: No Prediction")

chart = (
    alt
    .Chart(df_pred)
    .mark_line(point=True, size=1)
    .encode(
        alt.X('year:Q',
              scale=alt.Scale(padding=1),
              axis=alt.Axis(tickMinStep=1)),
        alt.Y('pred:Q',
              scale=alt.Scale(zero=False, padding=20, domainMax=1),
              axis=alt.Axis(format='%')),
        color='speed:O',
    )
)

st.altair_chart(chart, use_container_width=True)

st.caption(f"You can support me by adding a :star2: on the GitHub Page. It'll boost my analytics \:D")
# st.info(f"Performed {len(df_pred)} predictions")

with st.sidebar:
    st.markdown("""
    ![R2](https://img.shields.io/badge/R%20Squared-81.48%25-blueviolet)
    ![MAE](https://img.shields.io/badge/MAE-1.18%25-blue)
    ![RMSE](https://img.shields.io/badge/RMSE-1.71%25-blue)
    """)
    st.header("Requirements")
    st.markdown("""
    1) Only osu!mania.
    
    The user must be:
    1) ranked <10K in 1st Jan 2023
    2) active in that predicted year
    
    The map must be:
    1) ranked or loved
    2) played often enough 
    
    """)
    st.warning(":warning: Players and Maps that **barely** meet these may have incorrect predictions")
