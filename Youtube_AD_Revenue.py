import streamlit as st
import pandas as pd
import pickle
from datetime import datetime
import duckdb
import matplotlib.pyplot as plt
import seaborn as sns

model=pickle.load(open(r"model.pkl",'rb'))
scaler=pickle.load(open(r"scale.pkl",'rb'))
feature_names=pickle.load(open(r"feature_names.pkl",'rb'))

df=pd.read_csv(r"/Users/suriya/Ukesh_AIML_Projects/Content_Monetization_Modeler/Data/youtube_ad_revenue_dataset.csv")
df.drop_duplicates(inplace=True)
df['likes']=df['likes'].fillna(df['likes'].mean())
df['comments']=df['comments'].fillna(df['comments'].mean())
df['watch_time_minutes']=df['watch_time_minutes'].fillna(df['watch_time_minutes'].mean())

duckdb.register("youtube_data",df)

st.set_page_config(
    page_title="YouTube Ad Revenue",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)


col1,col2=st.columns([10,2])

with col1:
    st.title("📺 YouTube Ad Revenue Prediction")

with col2:
    df=pd.read_csv(r"/Users/suriya/Ukesh_AIML_Projects/Content_Monetization_Modeler/Data/youtube_ad_revenue_dataset.csv")
    dataset=df.to_csv(index=False)
    st.download_button(
        label="📥 Download Dataset",
        data=dataset,
        file_name="Dataset.csv",
        mime="text/csv",
        use_container_width=True
    )

    
st.sidebar.header("Enter Video Details")


tab1,tab2=st.tabs(['Prediction','Analytics'])

with tab1:

    views=st.sidebar.number_input("Views",0)
    likes=st.sidebar.number_input("Likes",0)
    comments=st.sidebar.number_input("Comments",0)
    watch_time=st.sidebar.number_input("Watch Time",0.0)
    video_length=st.sidebar.number_input("Video Length",0.0)
    subcribers=st.sidebar.number_input("Subscibers",0)
    category=st.sidebar.selectbox(
        "Category",
        ["Education","Music","Tech","Entertainment","Gaming","Lifestyle"]
        )
    device=st.sidebar.selectbox(
        "Device",
        ["TV","Mobile","Tablet","Desktop"]
        )
    country=st.sidebar.selectbox(
        "Country",
        ['IN','CA','UK','US','DE','AU']
        )

    date=st.sidebar.date_input("Upload_Date")
    time=st.sidebar.time_input("Upload_Time")

    st.markdown("""*As video creators and media companies increasingly rely on platforms like YouTube for income, predicting potential ad revenue has become essential for business planning and content strategy. This application uses a Machine Learning model to accurately estimate YouTube ad revenue based on various performance and contextual features.*""")

    st.info(" **Please Provide Video Details And Press Below Button** ")

    if st.button("Predict Revenue"):
        
        dt=datetime.combine(date,time) 
        
        sample=pd.DataFrame(0,index=[0],columns=feature_names)

        sample['views']=views
        sample['likes']=likes
        sample['comments']=comments
        sample['watch_time_minutes']=watch_time
        sample['video_length_minutes']=video_length
        sample['subscribers']=subcribers
        
        sample['Year']=dt.year
        sample['Month']=dt.month
        sample['Day']=dt.day
        sample['Hour']=dt.hour
        sample['Minute']=dt.minute
        sample['Seconds']=dt.second
        
        sample['engagement_rate'] = (sample['likes'] + sample['comments'])/(sample['views'] + 1)
        sample['likes_ratio'] = (sample['likes'])/(sample['views'] + 1)
        
        if f"category_{category}" in sample.columns:
            sample[f"category_{category}"]=1

        if f"device_{device}" in sample.columns:
            sample[f"device_{device}"]=1

        if f"country_{country}" in sample.columns:
            sample[f"country_{country}"]=1

        numeric_cols=[
            "views",
            "likes",
            "comments",
            "watch_time_minutes",
            "video_length_minutes",
            "subscribers",
            "engagement_rate",
            "likes_ratio"]

        sample[numeric_cols] = scaler.transform(sample[numeric_cols])
        
        prediction = model.predict(sample)
        
        st.divider()
        st.subheader("Prediction Result")
        
        col1,col2=st.columns(2)
        
        with col1:
            st.metric(label="Estimated Revenue",value=f"${prediction[0]:,.2f}")
        

        st.success("Prediction Completed Successfully")



watch_time_query="select floor(watch_time_minutes/10)*10 as watch_time_range, avg(ad_revenue_usd) as avg_revenue from youtube_data group by watch_time_minutes order by watch_time_minutes"

watch_time_df=duckdb.sql(watch_time_query).df()

country_query="select country, sum(ad_revenue_usd) as total from youtube_data group by country order by total"

country_df=duckdb.sql(country_query).df()


with tab2:
    st.subheader("Average Revenue by Watch Time")
    st.caption("X-axis: Watch Time (Minutes) | Y-axis: Average Revenue (USD)")
    st.bar_chart(watch_time_df.set_index("watch_time_range"))
    
    col1,col2=st.columns(2)
    
    with col1:
        st.subheader("Revenue by Country")
        st.caption("X-axis: Country | Y-axis: Revenue (USD)")
        st.bar_chart(country_df.set_index("country"))
    
    with col2:
        numeric_df = df.select_dtypes(include=['number'])
        corr = numeric_df.corr()
        fig, ax = plt.subplots(figsize=(12, 8))

        sns.heatmap(
            corr,
            annot=True,
            cmap="coolwarm",
            fmt=".2f",
            linewidths=0.5,
            ax=ax
        )

        ax.set_title("Correlation Heatmap")
        st.pyplot(fig)
        