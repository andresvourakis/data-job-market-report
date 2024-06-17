#######################################
# IMPORT LIBRARIES
#######################################

import streamlit as st
import pandas as pd
import json
import plotly.express as px
import math
import pickle
import nltk
from nltk.stem import WordNetLemmatizer
from collections import Counter
import re
from collections import defaultdict
import os
import plotly.graph_objects as go
import random

# Download NLTK data
nltk.download('wordnet')
nltk.download('omw-1.4')

#######################################
# PAGE SETUP
#######################################

# Set page configuration
st.set_page_config(
    page_title="Data Job Market Insights",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded",
)

#######################################
# DATA LOADING
#######################################

data_directory = './data'

# Load the pre-processed data
pickled_file_paths = ['keyword_categories.pkl', 'keyword_group_patterns.pkl', 'keyword_variation_patterns.pkl']
lemmatizer = WordNetLemmatizer()

# Function to open pickled files and return their contents
def open_pickled_files(file_paths):
    pickled_data = {}
    for file in file_paths:
        file_path = os.path.join(data_directory, file)
        with open(file_path, 'rb') as f:
            pickled_data[file] = pickle.load(f)
    return pickled_data

# Load pickled data
pickled_data = open_pickled_files(pickled_file_paths)

# Access specific data from the pickled files if needed
keyword_categories = pickled_data['keyword_categories.pkl']
keyword_group_patterns = pickled_data['keyword_group_patterns.pkl']
keyword_variation_patterns = pickled_data['keyword_variation_patterns.pkl']

# Read job descriptions
file_path = os.path.join(data_directory, 'job_descriptions.csv')
job_descriptions_df = pd.read_csv(file_path)

#######################################
# SIDEBAR FILTERS
#######################################

# Sidebar filters
st.sidebar.header("Filters")
experience_level = job_descriptions_df['experience_level_formatted'].unique()
title = job_descriptions_df['title_formatted'].unique()

selected_title = st.sidebar.selectbox("Job Title", title, index=0)
selected_experience_level = st.sidebar.multiselect("Experience Level", experience_level, default=experience_level[0])

# Apply filters
filtered_df = job_descriptions_df[job_descriptions_df['experience_level_formatted'].isin(selected_experience_level)]
filtered_df = filtered_df[filtered_df['title_formatted'] == selected_title]

st.sidebar.markdown("---")

#######################################
# PROCESS DATA
#######################################

# Process Keywords
def find_topics(job_ads, keyword_patterns):
    # Initialize a counter for keyword occurrences
    keyword_counter = Counter()
    
    # Lemmatize each word in the job ads and check for the presence of lemmatized keyword patterns
    for ad in job_ads:
        lemmatized_ad_words = [lemmatizer.lemmatize(word.lower()) for word in ad.split()]
        lemmatized_ad = ' '.join(lemmatized_ad_words)
        for first_keyword, pattern in keyword_patterns.items():
            if re.search(pattern, lemmatized_ad, re.IGNORECASE):  # Case insensitive matching
                keyword_counter[first_keyword] += 1
    
    # Sort the counter by value in descending order and return
    sorted_keyword_counter = Counter(dict(sorted(keyword_counter.items(), key=lambda item: item[1], reverse=True)))
    
    return sorted_keyword_counter

# Aggregate by category
def aggregate_counts_by_category(keyword_counter, keyword_categories):
    category_counts = defaultdict(dict)
    for category, keywords in keyword_categories.items():
        for keyword in keywords:
            if keyword in keyword_counter:
                category_counts[category][keyword] = keyword_counter[keyword]
    return category_counts

# Data Types
# Ensure the 'date' column is in datetime format
filtered_df['date'] = pd.to_datetime(filtered_df['date'], errors='coerce')

# Filter data
filtered_job_descriptions = filtered_df['description'].tolist()
total_filtered_jobs = len(filtered_job_descriptions)
min_date_filtered = filtered_df['date'].min()
max_date_filtered = filtered_df['date'].max()

# Calculate counters
keyword_group_count = find_topics(filtered_job_descriptions, keyword_group_patterns)
keyword_variation_count = find_topics(filtered_job_descriptions, keyword_variation_patterns)
keyword_category_count = aggregate_counts_by_category(keyword_variation_count, keyword_categories)

#######################################
# VISUALIZATION METHODS
#######################################

def get_date_range(min_date, max_date):

    date_string = ""

    # Extract month and year parts
    min_month = min_date.strftime('%B')
    max_month = max_date.strftime('%B')
    min_year = min_date.strftime('%Y')
    max_year = max_date.strftime('%Y')

    # Construct the string based on year equality
    if min_year == max_year:
        date_string = f"{min_month}-{max_month}, {min_year}"
    else:
        date_string = f"{min_month}, {min_year} - {max_month}, {max_year}"
    
    return date_string

def plot_metric(label, value, prefix="", suffix="", show_graph=False, color_graph=""):
    fig = go.Figure()

    fig.add_trace(
        go.Indicator(
            value=value,
            gauge={"axis": {"visible": True}},
            number={
                "prefix": prefix,
                "suffix": suffix,
                "font.size": 38,
                "font": {'color': "black"}
            },
            title={
                "text": f"<span style='color: black;'>{label}</span>",
                "font": {"size": 24},
            },
        )
    )
    fig.update_layout(
        paper_bgcolor="#3D8693",
        margin=dict(t=30, b=0),
        showlegend=False,
        plot_bgcolor="white",
        height=100,
    )

    st.plotly_chart(fig, use_container_width=True)

# Plot categories
# Visualize the data as percentages of total job ads
def visualize_percent_job_total(keyword_count, total_job_ads, top, title):
    keywords, values = zip(*keyword_count.items())
    
    percentages = [value / total_job_ads * 100 for value in values]
    
    # Create a DataFrame for Plotly
    df = pd.DataFrame({
        'Keyword': keywords,
        'Percentage': percentages
    })
    
    # Plot with Plotly
    fig = px.bar(df.head(top), x='Percentage', y='Keyword', orientation='h', color='Keyword',
                 labels={'Percentage': 'Percentage (%)', 'Keyword': 'Keywords'},
                 title=f'{title}',
                 text='Percentage')
    
    # Remove legend
    fig.update_layout(showlegend=False)
    
    # Update text position and format
    fig.update_traces(texttemplate='%{text:.0f}%', textposition='outside')
    
    st.plotly_chart(fig, use_container_width=True)


def visualize_category_percent_job_total(category_counts, total_job_ads):
    # Create two columns
    left_column, right_column = st.columns(2)
    
    for index, (category, counts) in enumerate(category_counts.items()):
        sorted_counts = sorted(counts.items(), key=lambda item: item[1], reverse=True)
        keywords, values = zip(*sorted_counts)
        
        percentages = [(value / total_job_ads) * 100 for value in values]
        
        # Create a DataFrame for Plotly
        df = pd.DataFrame({
            'Keyword': keywords,
            'Percentage': percentages
        })
        
        # Plot with Plotly
        fig = px.bar(df, x='Percentage', y='Keyword', orientation='h', color='Keyword',
                     labels={'Percentage': 'Percentage (%)', 'Keyword': 'Keywords'},
                     title=f'{category}')
        
        # Remove legend
        fig.update_layout(showlegend=False)

        # Update text position and format
        fig.update_traces(texttemplate='%{x:.0f}%', textposition='outside')
        
        # Assign the visualization to the appropriate column
        if index % 2 == 0:
            with left_column:
                st.plotly_chart(fig)
        else:
            with right_column:
                st.plotly_chart(fig)

def visualize_spider_chart(category_counts, title):
    categories = list(category_counts.keys())

    # Aggregate the total counts for each category
    category_totals = {category: sum(counts.values()) for category, counts in category_counts.items()}

    # Create a DataFrame for Plotly
    df = pd.DataFrame({
        'Category': list(category_totals.keys()),
        'Total': list(category_totals.values())
    })

    # Create radar chart
    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=df['Total'],
        theta=df['Category'],
        fill='toself',
        name='Category Totals'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(df['Total'])]
            )),
        showlegend=False,
        title=title
    )

    # Display the chart in Streamlit
    st.plotly_chart(fig)

# Custom footer HTML
custom_footer = """
    <style>
    #custom-footer {
        position: relative;
        background-color: #3D8693;
        color: white;
        text-align: center;
        padding: 0px;
        margin-top: 10px;
    }
    #custom-footer p {
        margin: 3px 0;  /* Adjust margin to reduce space between lines */
        padding: 0;
    }
    #custom-footer a {
        color: #FFD700 !important;  /* Change hyperlink color here (yellow) */
        text-decoration: none;
    }
    #custom-footer a:hover {
        color: #FFFF00;  /* Optional: Change color on hover (gold) */
        text-decoration: underline;
    }
    </style>
    <footer id="custom-footer">
        <br>
        <p>Created by Andres Vourakis – Connect with me on 
        <a href="https://www.linkedin.com/in/andresvourakis/" target="_blank">
            <img src="https://img.icons8.com/ios-filled/50/ffffff/linkedin.png" width="20"/>
        </a>
        <p>This resource is part of 
        <a href="https://course.andresvourakis.com/ds-hire-ready" target="_blank">
            Data Science Hire Ready
        </a></p>
        <br> <!-- Add an empty line of text -->
    </footer>
    """

# Hide the "Made with Streamlit" footer
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    #custom-footer {visibility: visible;}
    </style>
    """

# Function to display the nested dictionary as an expandable table with percentages, sorted by count, and ranked
def display_expandable_table_with_percentages(nested_dict):
    for category, skills in nested_dict.items():
        with st.expander(category):
            total = total_filtered_jobs #sum(skills.values())
            data = [{'Skill': skill, 'Count': count, 'Percent of Total': f"{(count / total) * 100}"} for skill, count in skills.items()]
            df = pd.DataFrame(data)
            df = df.sort_values(by='Count', ascending=False)  # Sort by Count
            df.insert(0, 'Rank', range(1, len(df) + 1))  # Add Rank column starting from 1
            df = df.reset_index(drop=True)  # Reset the index and drop the original index

            # Display dataframe as table
            st.dataframe(df,
                column_order=("Rank", "Skill", "Count", "Percent of Total"),
                hide_index=True,
                width=1000,
                column_config={
                    "Percent of Total": st.column_config.ProgressColumn(
                    "Percent of Total",
                    format="%.0f%%",
                    min_value=0,
                    max_value=100,
                )}    
            )

# Function to display the nested dictionary as expandable tables with percentages, sorted by count, and ranked
def display_expandable_table_with_percentages(nested_dict, total_filtered_jobs):
    # Create two columns
    col1, col2 = st.columns(2)
    
    # Iterate through the categories and skills
    for i, (category, skills) in enumerate(nested_dict.items()):
        total = total_filtered_jobs
        data = [{'Skill': skill, 'Count': count, 'Percent of Total': f"{(count / total) * 100}"} for skill, count in skills.items()]
        df = pd.DataFrame(data)
        df = df.sort_values(by='Count', ascending=False)  # Sort by Count
        df.insert(0, 'Rank', range(1, len(df) + 1))  # Add Rank column starting from 1
        df = df.reset_index(drop=True)  # Reset the index and drop the original index

        # Alternate placing the expanders in the two columns
        if i % 2 == 0:
            with col1.expander(category):
                st.dataframe(
                    df,
                    column_order=("Rank", "Skill", "Count", "Percent of Total"),
                    hide_index=True,
                    width=1000,
                    column_config={
                        "Percent of Total": st.column_config.ProgressColumn(
                            "Percent of Total",
                            format="%.0f%%",
                            min_value=0,
                            max_value=100,
                        )
                    }
                )
        else:
            with col2.expander(category):
                st.dataframe(
                    df,
                    column_order=("Rank", "Skill", "Count", "Percent of Total"),
                    hide_index=True,
                    width=1000,
                    column_config={
                        "Percent of Total": st.column_config.ProgressColumn(
                            "Percent of Total",
                            format="%.0f%%",
                            min_value=0,
                            max_value=100,
                        )
                    }
                )


#######################################
# STREAMLIT LAYOUT
#######################################

st.title(":bar_chart: Job Market Report")
st.subheader(f"_{selected_title} ({','.join(selected_experience_level)})_")

with st.container():
    st.markdown("### Overview")
    st.markdown("Overview of which type of relevent keywords are present in job ads")

    top_left_column, top_right_column = st.columns(2)  # Adjust column width

    with top_left_column:
        visualize_percent_job_total(keyword_group_count, total_filtered_jobs, 10, "Top Skills")

    with top_right_column:
        visualize_spider_chart(keyword_category_count, "Distribution of Skils")

with st.container():
    st.markdown("### Top Skills per Category")
    st.markdown("Relevant keywords grouped into common categories")
    display_expandable_table_with_percentages(keyword_category_count, total_filtered_jobs)

# For filter secition
st.sidebar.markdown(
    f'''
    ### Data Overview
    *Based on selected filters.*
    - **Data Source**: LikedIn Jobs.
    - **Total Job Ads**: :red[{total_filtered_jobs}]
    - **Date Range**: :red[{get_date_range(min_date_filtered, max_date_filtered)}]
    - **Update Frequency**: Weekly
    - **Jobs Location**: United States
    ''')

# Hide default Streamlit app
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Embed custom footer in Streamlit app
st.markdown(custom_footer, unsafe_allow_html=True)
