import pandas as pd
import streamlit as st

st.set_page_config("Info", page_icon=":question:")

st.title("Info")

r2 = 0.7376
mae = 0.0109
rmse = 0.0162
version = "May%202023"
st.header(":game_die: Metrics")
st.markdown(f"""
![R2](https://img.shields.io/badge/R2-{r2:.2%}-blueviolet)
![MAE](https://img.shields.io/badge/MAE-{mae:.2%}-blue)
![RMSE](https://img.shields.io/badge/RMSE-{rmse:.2%}-blue)
![Version](https://img.shields.io/badge/V3-{version}-purple)
![Model Size](https://img.shields.io/github/size/Eve-ning/opal/opal/models/V3_2023_05/model.ckpt)
""")

st.header(":bookmark: Requirements")


st.markdown("""
- Only osu!mania.

The user must be:
- ranked <10K in 1st Apr 2023
- active in that predicted year

The map must be:
- ranked or loved
- played often enough 
- little in SVs
- \>2.0 SR
""")

st.warning(":warning: Players and Maps that **barely** meet these may have incorrect predictions")
st.header(":question: FAQ")
c1, c2 = st.columns([2, 3])

with c1:
    st.write(":question: Why is my predicted HT score worse / DT Score better?")
with c2:
    st.write(":information_source: "
             "This happens when there's too little associations between players that play these different mods. "
             "For example, if there are 2 independent groups of players that play each mod, it'll be impossible to "
             "tell which mod is harder!")

st.markdown("---")
c1, c2 = st.columns([2, 3])

with c1:
    st.write(":question: Why does the algorithm underestimate my performance?")
with c2:
    st.write(":information_source: "
             "One of the main reasons is because we weigh judgements differently. "
             "In osu!mania, 300s and MAX/300G both weigh 1.0. "
             "In opal, we weigh MAX as 320/320, 300s as 300/320, 200s as 200/320, and so on.")
    st.table(
        pd.DataFrame(
            {
                'osu!mania': map(lambda x: f"{x:.0%}", [1.0, 1.0, 2 / 3, 1 / 3, 1 / 6, 0]),
                'opal': map(lambda x: f"{x:.0%}", [1.0, 300 / 320, 200 / 320, 100 / 320, 50 / 320, 0]),
            }, index=['300G/MAX', '300', '200', '100', '50', '0']
        )
    )

st.markdown("---")
c1, c2 = st.columns([2, 3])

with c1:
    st.markdown(":question: Why did the prediction fail?")
with c2:
    st.markdown("""
    :information_source: This can happen when 
    1) You're not in the Top 10K Overall rank By 1st Jan 2023
    2) The map you predicted wasn't played enough
    3) You didn't play enough for that year 
    """)

st.markdown("---")
c1, c2 = st.columns([2, 3])

with c1:
    st.markdown(":question: How does it work?")
with c2:
    st.markdown("""
    :information_source: We find associations between player scores to predict your performances.
    
    It uses a Collaborative Filtering Algorithm, which is widely used in Recommendation. 
    For example, when shopping items are recommended to you, it's because you matched similar interests to customers
    who have similar buying tastes.
    
    In this case, we're not recommending items, instead, finding similar players to you, then aggregating their scores
    to find a suitable estimate to your performance!  
    """)

st.markdown("---")
c1, c2 = st.columns([2, 3])

with c1:
    st.markdown(":question: Can I use it in my programming projects?")
with c2:
    st.markdown("""
    :information_source: Yes! However, I highly recommend to download the model locally to use in PyTorch Lightning.
    
    You can find this model and its usage instructions in my GitHub.
    See the main page and click the GitHub icon
    
    This brings many benefits  
    1) It's multiple times faster as it's not bottle-necked by this UI
    2) You have a static, local copy. That means, if I do update it, the result shouldn't change
    
    If you do use it in a paper, in your own game, or anything similar like a school project, please credit me > w<)b.
    """)

st.markdown("---")
c1, c2 = st.columns([2, 3])

with c1:
    st.markdown(":question: Where can I feedback?")
with c2:
    st.markdown("""
    :information_source:
    You can feedback to me on the [Opal GitHub Repo](https://github.com/Eve-ning/opal). Open an issue and write away!
    
    It doesn't have to be professionally written! Just keep your issue short and to the point o wo)b
    
    If you're unsure, feel free to ping me on [Twitter](https://twitter.com/dev_evening).
    """)
