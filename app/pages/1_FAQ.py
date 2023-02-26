import streamlit as st

st.set_page_config("FAQ", page_icon=":question:")

st.title("FAQ")
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
