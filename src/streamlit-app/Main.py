import base64
import json
import os
import sys
from pathlib import Path

from google.cloud import firestore

sys.path.append(Path(__file__).parents[1].as_posix())
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import altair as alt
import numpy as np
import pandas as pd
import requests
import streamlit as st

from opal import OpalNet

st.set_page_config(page_title="Opal | o!m AI Score Predictor", page_icon=":comet:")
if 'predictions' not in st.session_state:
    st.session_state['predictions'] = 0

osu_api_key = st.secrets['OSU_API']
fb_api_key = st.secrets['FB_KEY']


@st.cache_resource
def get_db_pred_ref():
    print("Initializing Firebase Reference")
    key_b64_decoded = base64.b64decode(fb_api_key)
    db = firestore.Client.from_service_account_info(json.loads(key_b64_decoded))

    return db.collection("predictions").document("predict")


db_pred = get_db_pred_ref()


def add_analytics_count(add: int):
    """ Adds `add` amount of 'predictions' into the firebase count """
    count = db_pred.get().to_dict()['count']
    db_pred.update({'count': count + add})
    st.session_state['predictions'] += add


@st.cache_resource
def get_model():
    return OpalNet.load(map_location='cpu')


def random_uid():
    st.session_state['uid'] = np.random.choice(net.uid_le.classes_).split("/")[0]


def random_mid():
    st.session_state['mid'] = np.random.choice(net.mid_le.classes_).split("/")[0]


def predict(uid, mid):
    try:
        preds = float(net.predict(uid, mid))
        return preds
    except ValueError:
        return None


@st.cache_data
def get_user_id(username: str):
    url = f'https://osu.ppy.sh/api/get_user?k={osu_api_key}&u={username}&type=string'

    response = requests.get(url)

    try:
        user_data = response.json()[0]
        user_id = user_data['user_id']
        return int(user_id)
    except:
        return


@st.cache_data
def get_username(user_id: int):
    url = f'https://osu.ppy.sh/api/get_user?k={osu_api_key}&u={user_id}'

    response = requests.get(url)

    try:
        user_data = response.json()[0]
        username = user_data['username']
        return username
    except:
        return


@st.cache_data
def get_beatmap_metadata(beatmap_id: int):
    url = f'https://osu.ppy.sh/api/get_beatmaps?k={osu_api_key}&b={beatmap_id}'

    response = requests.get(url)

    try:
        map_data = response.json()[0]
        map_metadata = f"{map_data['artist']} - " \
                       f"{map_data['title']} (" \
                       f"{map_data['creator']}) [" \
                       f"{map_data['version']}]"
        return map_metadata
    except:
        return None


MODEL_YEAR = 2023
YEARS_SEARCH = 10
DEFAULT_USER_ID = "2193881"
DEFAULT_MAP_ID = "767046"
net = get_model()
net = net.to('cpu')

with st.sidebar:
    st.info("""
    **Judgements are out of 320, thus we underestimate.**\n\nSee FAQ (2) for more info 
    """)
    st.warning("""
    :warning: If HT < NT or DT > NT, it's due to the modded plays not having enough samples.  
    """)
    st.header(":wave: Hey! [Try AlphaOsu!](https://alphaosu.keytoix.vip/)")
    st.caption("AlphaOsu is a pp recommender system with a website UI. ")
    st.caption("Opal doesn't require monetary support, but they do. "
               "If you do enjoy using their services, "
               "you can [support them](https://alphaosu.keytoix.vip/support)")

st.markdown(f"""
<h1 style='text-align: center;'>
<span style='filter: drop-shadow(0 0.2mm 1mm rgba(142, 190, 255, 0.9));'>Opal</span>\n

<p style='color:grey'>AI Score Predictor by 
    <a href='https://github.com/Eve-ning/' style='text-decoration:none'>Evening</a>
    
<a href='https://twitter.com/dev_evening' style='text-decoration:none'>![Twitter](https://img.shields.io/badge/-dev__evening-blue?logo=twitter)</a>
<a href='https://github.com/Eve-ning/opal' style='text-decoration:none'>![Repo](https://img.shields.io/badge/Repository-purple?logo=github)</a>
![Predictions](https://img.shields.io/badge/Predictions-{db_pred.get().to_dict()['count']:,}-yellow?logo=firebase)
</p>
</h1>
""", unsafe_allow_html=True)

with st.container():
    left, right = st.columns(2)
    with left:
        st.button("Get Random Player", on_click=random_uid)
        uid_username = st.text_input("User ID/Username", key='uid', placeholder="Evening")
    with right:
        st.button("Get Random Map", on_click=random_mid)
        mid = st.text_input("Map ID", key='mid', placeholder="767046")

    try:
        try:
            uid = int(uid_username)
            username = get_username(uid)
        except ValueError:
            username = uid_username
            uid = get_user_id(username)

        mid = int(mid)
        map_metadata = get_beatmap_metadata(mid)
        left.markdown(f"<a href='https://osu.ppy.sh/users/{uid}' style='text-decoration:none'>{username}</a>",
                      unsafe_allow_html=True)
        right.markdown(f"<a href='https://osu.ppy.sh/b/{mid}' style='text-decoration:none'>{map_metadata}</a>",
                       unsafe_allow_html=True)
    except:
        st.stop()

# Prediction Logic
with st.container():
    years = range(MODEL_YEAR - YEARS_SEARCH, MODEL_YEAR + 1)
    speeds = {-1: 'HT', 0: 'NT', 1: 'DT'}

    preds = []

    for speed, speed_txt in speeds.items():
        for year in years:
            uid_ = f"{uid}/{year}"
            mid_ = f"{mid}/{speed}"
            if uid_ in net.uid_le.classes_ and mid_ in net.mid_le.classes_:
                pred = predict(uid_, mid_)
                if pred:
                    preds.append([year, speed_txt, pred])

    df_pred = pd.DataFrame(preds, columns=['year', 'speed', 'pred'])
    df_pred_last = df_pred[df_pred['year'] == df_pred['year'].max()]

    pred_ht = df_pred_last.loc[(df_pred_last['speed'] == 'HT'), 'pred']
    pred_nt = df_pred_last.loc[(df_pred_last['speed'] == 'NT'), 'pred']
    pred_dt = df_pred_last.loc[(df_pred_last['speed'] == 'DT'), 'pred']

    pred_ht = float(pred_ht.iloc[0]) if pred_ht.any() else None
    pred_nt = float(pred_nt.iloc[0]) if pred_nt.any() else None
    pred_dt = float(pred_dt.iloc[0]) if pred_dt.any() else None

    c1, c2, c3 = st.columns(3)

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

    add_analytics_count(len(df_pred))

# Chart Logic
with st.container():
    chart = (
        alt
        .Chart(df_pred)
        .mark_line(point=True, size=1)
        .encode(
            alt.X('year:Q',
                  scale=alt.Scale(padding=1),
                  axis=alt.Axis(tickMinStep=1),
                  title="Year"),
            alt.Y('pred:Q',
                  scale=alt.Scale(zero=False, padding=20, domainMax=1),
                  axis=alt.Axis(format='%'),
                  title="Predicted Accuracy"),
            color='speed:O'
        )
    )

    st.altair_chart(chart, use_container_width=True)

st.caption(f"You can support me by adding a :star2: on the "
           f"<a href='https://github.com/Eve-ning/opal' style='text-decoration:none'>GitHub Page</a>. "
           f"It'll boost my analytics > w<)9!",
           unsafe_allow_html=True)

if st.session_state['predictions']:
    st.caption(f"You've contributed **{st.session_state['predictions']}** predictions to my analytics, thanks!")

st.caption(":grey_exclamation: *We do not track the history of your predictions, only the count*")