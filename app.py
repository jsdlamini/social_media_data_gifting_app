import streamlit as st
import pandas as pd
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from textblob import TextBlob 
import json
import nltk
import datetime
import regex
import altair as alt
import streamlit as st  
from streamlit_extras.metric_cards import style_metric_cards 
from streamlit_extras.colored_header import colored_header 
import plotly.graph_objects as go
import plotly.express as px
from  streamlit_option_menu import option_menu
import os 
# from datetime import datetime 
from google.colab import files 
# Define English stopwords
import pandas as pd
from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
from nltk import word_tokenize , sent_tokenize
import nltk
import json 
import string 
#######################
# Page configuration
# Set up the page config to set the title of the app 
st.set_page_config(
    page_title="Social Media Data Gifting Tool",
    page_icon="🏂",
    layout="wide",
    initial_sidebar_state="expanded")

alt.themes.enable("dark")

 
st.header("Social Media Data Gifting Tool")
#######################
#all graphs we use custom css not streamlit 
theme_plotly = None 


# load Style css
with open('style.css')as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html = True)


     
nltk.download('punkt')
nltk.download('stopwords')

def preprocess_texts(texts):
    stop_words = set(stopwords.words('english'))
    cleaned_texts = []
    for text in texts:
        words = word_tokenize(text.lower())
        filtered_words = [word for word in words if word.isalnum() and word not in stop_words]
        cleaned_texts.append(' '.join(filtered_words))
    return cleaned_texts 
    
  
def analyze_sentiment(texts):
    avg_polarity = 0.0 
    sentiments = []
    for text in texts:
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity 
        sentiments.append(polarity)
    avg_polarity = sum(sentiments) / len(sentiments) if sentiments else 0.0
    return avg_polarity 

# Function to analyze sentiment
def analyze_comment_sentiment(comment):
            return TextBlob(comment).sentiment.polarity  # Returns a value between -1 (negative) and 1 (positive)
        
    
def generate_word_cloud(text):
    wordcloud = WordCloud(width=400, height=200, background_color='white').generate(text)
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)
  
def generate_bar_chart(word_counts):
    words, counts = zip(*word_counts)
    fig, ax = plt.subplots(figsize=(4, 2))
    #  bars = ax.bar(categories, values, color='skyblue', edgecolor='black')
    ax.bar(words, counts, color='skyblue', edgecolor='black')
    # ax.set_title('Top Words by Frequency')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_ylabel('Frequency')
    ax.set_xticklabels(words, rotation=45, ha='right')
    st.pyplot(fig)




def calculate_metrics(comments):
    lengths = [len(comment.split()) for comment in comments]
    avg_length = sum(lengths) / len(lengths) if lengths else 0
    max_length = max(lengths) if lengths else 0
    return avg_length, max_length

def plot_metrics(avg_length, max_length):
    metrics = ['Avg Length', 'Max Length']
    values = [avg_length, max_length]
    
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(metrics, values, color=['skyblue', 'lightgreen', 'salmon'])
    ax.set_title('Comment Metrics')
    st.pyplot(fig)

def create_donut_chart(percentage, color='red'):
    # Define figure and axis for plot
    fig, ax = plt.subplots(figsize=(1, 1))
    # Donut chart data 
    sizes = [percentage, 100 - percentage]
    wedges, texts, autotexts = ax.pie(sizes, labels=['', ''], startangle=90,
                                      colors=[color, 'lightgray'], 
                                      autopct='', pctdistance=0.85,
                                      wedgeprops=dict(width=0.3)) 
    # Draw a circle at the center for donut hole effect 
    center_circle = plt.Circle((0, 0), 0.70, fc='white')
    fig.gca().add_artist(center_circle)
    
    # Ensure aspect ratio is equal for a nice circle
    ax.axis('equal')
    # Reduce the margin
    ax.margins(0.05, 0.05)    
    # Label the donut with the percentage
    plt.text(0, 0, f'{percentage:.1f}%', ha='center', va='center', fontsize=6, color=color)
    # Apply tight layout
    plt.tight_layout()
    # Return the matplotlib figure
    return fig
 
def remove_stop_words(comment):
    stop_words = set(stopwords.words('english'))
    # Tokenize the comment
    words = word_tokenize(comment)
    # Remove stop words
    filtered_words = [word for word in words if word.lower() not in stop_words]
    # Rejoin the words into a single string
    return ' '.join(filtered_words)
    
def preprocess(text):
    # Tokenize and remove punctuation
    tokens = nltk.word_tokenize(text)
    tokens = [word.lower() for word in tokens if word.isalpha()]
    # Remove stopwords
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return ' '.join(tokens)
    
def process_comments(comments):
    words = ' '.join(comments).split()
    most_common = Counter(words).most_common(10)
    combined_text = ' '.join(words)
    return most_common, combined_text

def remove_punc(text): 
    punc=string.punctuation
    return text.translate(str.maketrans('', '',punc))
     
def replace_abbreviations(text):
    abbreviation_dict = {
    'LOL': 'laugh out loud',
    'BRB': 'be right back',
    'OMG': 'oh my god',
    'AFAIK': 'as far as I know',
    'AFK': 'away from keyboard',
    'ASAP': 'as soon as possible',
    'ATK': 'at the keyboard',
    'ATM': 'at the moment',
    'A3': 'anytime, anywhere, anyplace',
    'BAK': 'back at keyboard',
    'BBL': 'be back later',
    'BBS': 'be back soon',
    'BFN': 'bye for now',
    'B4N': 'bye for now',
    'BRB': 'be right back',
    'BRT': 'be right there',
    'BTW': 'by the way',
    'B4': 'before',
    'B4N': 'bye for now',
    'CU': 'see you',
    'CUL8R': 'see you later',
    'CYA': 'see you',
    'FAQ': 'frequently asked questions',
    'FC': 'fingers crossed',
    'FWIW': 'for what it\'s worth',
    'FYI': 'For Your Information',
    'GAL': 'get a life',
    'GG': 'good game',
    'GN': 'good night',
    'GMTA': 'great minds think alike',
    'GR8': 'great!',
    'G9': 'genius',
    'IC': 'i see',
    'ICQ': 'i seek you',
    'ILU': 'i love you',
    'IMHO': 'in my honest/humble opinion',
    'IMO': 'in my opinion',
    'IOW': 'in other words',
    'IRL': 'in real life',
    'KISS': 'keep it simple, stupid',
    'LDR': 'long distance relationship',
    'LMAO': 'laugh my a.. off',
    'LOL': 'laughing out loud',
    'LTNS': 'long time no see',
    'L8R': 'later',
    'MTE': 'my thoughts exactly',
    'M8': 'mate',
    'NRN': 'no reply necessary',
    'OIC': 'oh i see',
    'PITA': 'pain in the a..',
    'PRT': 'party',
    'PRW': 'parents are watching',
    'QPSA?': 'que pasa?',
    'ROFL': 'rolling on the floor laughing',
    'ROFLOL': 'rolling on the floor laughing out loud',
    'ROTFLMAO': 'rolling on the floor laughing my a.. off',
    'SK8': 'skate',
    'STATS': 'your sex and age',
    'ASL': 'age, sex, location',
    'THX': 'thank you',
    'TTFN': 'ta-ta for now!',
    'TTYL': 'talk to you later',
    'U': 'you',
    'U2': 'you too',
    'U4E': 'yours for ever',
    'WB': 'welcome back',
    'WTF': 'what the f...',
    'WTG': 'way to go!',
    'WUF': 'where are you from?',
    'W8': 'wait...',
    '7K': 'sick laughter',
    'TFW': 'that feeling when',
    'MFW': 'my face when',
    'MRW': 'my reaction when',
    'IFYP': 'i feel your pain',
    'LOL': 'laughing out loud',
    'TNTL': 'trying not to laugh',
    'JK': 'just kidding',
    'IDC': 'i don’t care',
    'ILY': 'i love you',
    'IMU': 'i miss you',
    'ADIH': 'another day in hell',
    'IDC': 'i don’t care',
    'ZZZ': 'sleeping, bored, tired',
    'WYWH': 'wish you were here',
    'TIME': 'tears in my eyes',
    'BAE': 'before anyone else',
    'FIMH': 'forever in my heart',
    'BSAAW': 'big smile and a wink',
    'BWL': 'bursting with laughter',
    'LMAO': 'laughing my a** off',
    'BFF': 'best friends forever',
    'CSL': 'can’t stop laughing',
    }
    for abbreviation, full_form in abbreviation_dict.items():
        text = text.replace(abbreviation, full_form)
    return text
  
def tokenize_text(text):
    # Tokenize each sentence into words
    words_list = [word_tokenize(sentence) for sentence in sent_tokenize(text)] 
    words = ' '.join(' '.join(words) for words in words_list) 
    return words

# remove html tags¶ 
## check if there is html tags 
def has_html_tags(text):
    soup = BeautifulSoup(text, 'html.parser')
    return bool(soup.find()) 


def has_emoji(text):
    
      # Incorrectly decoded bytes
     # Convert the string to bytes using 'latin1' as it can handle full 0-255 byte range without errors
    # byte_representation = text.encode('utf-8')

    # # Decode the byte representation using 'utf-8' to properly interpret emoji bytes
    # text = byte_representation.decode('unicode_escape')
    
    emoji_pattern = regex.compile(r'[\p{Emoji}\p{Emoji_Presentation}\p{Extended_Pictographic}]+', flags=regex.UNICODE)
    
  

    if bool(emoji_pattern.search(text)):
        st.write(text)
    return bool(emoji_pattern.search(text))


def remove_emojis(text):
    # Compile a pattern to match emojis using regex
    # emoji_pattern = regex.compile(r'\p{Emoji}', flags=regex.UNICODE)
    emoji_pattern= regex.compile(r'[\p{Emoji}\p{Emoji_Presentation}\p{Extended_Pictographic}]+', flags=regex.UNICODE)
    # Substitute emojis with an empty string
    return emoji_pattern.sub('', text)
    
    
def remove_non_english(text):
    # Define a regex pattern to keep only English letters, numbers, and basic punctuation
    pattern = regex.compile(r'[^\w\s,.!?\'"`~-]')
    # Substitute any character not matching the pattern with an empty string
    
    
    return pattern.sub('', text) 
      
def remove_malformed_emojis(text):
    # Pattern to identify malformed emoji sequences
    malformed_emoji_pattern = regex.compile(
        r'(?:(?:[\u00C0-\u00FF][\u0080-\u00BF])+)', flags=regex.UNICODE
    )
    
    # Substitute these sequences with an empty string
    cleaned_text = malformed_emoji_pattern.sub('', text)
    
    return cleaned_text
    
    
# Create a horizontal bar chart using Matplotlib
def plot_horizontal_bar_chart(data):
    # Set the figure size for a tiny chart
    plt.figure(figsize=(2, 1),  width=None)  # Width x Height in inches
    plt.barh(data.columns, data.iloc[0], color='red')  # Horizontal bar chart
    plt.xlabel('Values')
    plt.title('Average Comment Statistics')
    plt.xlim(0, max(data.max()) + 10)  # Adjust x-axis limits for better visibility

    # Show the plot in Streamlit
    st.pyplot(plt)
 
# Function to compute subjectivity for each comment
def get_subjectivity(text):
    return TextBlob(text).sentiment.subjectivity
     
def nice_header(label="My New Pretty Colored Header", description="This is a description", color_name="violet-70" ):
    colored_header(
        label=label ,
        description=description ,
        color_name=color_name,
    )




def donateData(folder):
        SAVE_DIR = folder
        
        # Ensure the directory exists
        if not os.path.exists(SAVE_DIR):
            os.makedirs(SAVE_DIR)
        
        # File uploader
        user_uploaded_files = st.file_uploader(
            "Choose one or more JSON files",
            key="data_donate",
            type="json",
            accept_multiple_files=True
        )
         
        if user_uploaded_files:
            for uploaded_file in user_uploaded_files:
                # Read file content into bytes
                file_content = uploaded_file.read()
                
                # Create a path for the new file
                  # Generate a timestamp and append to the original file name (excluding extension)
                file_name = uploaded_file.name
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                base_name, ext = os.path.splitext(file_name)
                
                unique_file_name = f"{base_name}_{timestamp}{ext}"
                        
                save_path = os.path.join(SAVE_DIR, unique_file_name )
                
                # Write the file content to a new file
                with open(save_path, 'wb') as out_file:
                    out_file.write(file_content)
                
                st.success(f"Saved file: {unique_file_name} to {save_path}")
        
def read_json_files_from_directory(directory="uploaded_files"):
    """
    Reads JSON files from a specified directory and returns data as Pandas DataFrames.

    Parameters:
    - directory (str): Path to the directory containing the JSON files.
 
    Returns:
    - list of tuples: A list where each tuple contains the filename and its corresponding DataFrame.
    """
    # Get list of all JSON files in the directory
    file_list = [f for f in os.listdir(directory) if f.endswith('.json')]

    dataframes = []  # This will store data from each JSON file as DataFrames

    if file_list:
        # Multi-select input box for file selection
        selected_files = st.multiselect('Select JSON file(s) to analyze', file_list)
        # st.write( "Files are ", selected_files)

        # Read and process selected JSON files
        for file_name in selected_files:
            file_path = os.path.join(directory, file_name)

            with open(file_path, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)

                    # Convert JSON data to DataFrame if possible
                    if isinstance(data, list) and all(isinstance(item, dict) for item in data):
                        df = pd.DataFrame(data)
                        dataframes.append((file_name, df))

                        # Displaying the data for each selected file
                        st.write(f"Data from {file_name}:")
                        st.dataframe(df)
                    # else:
                    #     st.warning(f"{file_name} does not contain a list of records suitable for conversion to a DataFrame.")
                except json.JSONDecodeError:
                    st.error(f"Error decoding {file_name}. File may not be valid JSON.")

    else:
        st.info('No JSON files found in the directory. Please upload files for processing.')

    return dataframes , selected_files
      
# from_date = selected_from_dates = st.sidebar.date_input("From: ", datetime.date(2019, 7, 6)) 
# to_date=selected_to_dates = st.sidebar.date_input("To:  ", datetime.date(2024, 10, 30))
 
# uploaded_file = st.sidebar.file_uploader("Choose json file", type="json")
 
 
with st.sidebar:
    selected = option_menu(
        menu_title="Main Menu",
        options=["Home", "Data Donation", "Data Analysis", "settings"] ,
        icons=['house','share','file-bar-graph', 'gear'], menu_icon="cast", default_index=1
        
        )

if selected =="Home":
    # st.title(f"Home"). 
    nice_header(label="Home", description="The Social Media Data Gifting Tool is an innovative application designed to leverage the wealth of textual data generated by Facebook users, specifically focusing on the rich insights encapsulated within user comments. By employing advanced Natural Language Processing (NLP) techniques, this tool meticulously analyzes each comment, uncovering patterns and sentiments that might otherwise go unnoticed. It offers a dual analytical approach: it can dissect the digital footprint of an individual student, providing personalized insights and feedback, as well as evaluate a larger body of work from an entire class, or sample group, thus delivering a more comprehensive overview of collective thoughts and trends. This bifurcated capability allows educators and researchers to adapt the tool's use according to their specific analytical needs—whether honing in on individual student behavior for tailored educational guidance or assessing broader learning outcomes and social dynamics across a group. By transforming raw comment data into actionable intelligence, the Social Media Data Gifting Tool opens new horizons for understanding and fostering positive educational and social environments.",  color_name="light-blue-70")
    
    #  "red-70". Supported colors are "light-blue-70", "orange-70", "blue-green-70", "blue-70", "violet-70", "red-70", "green-70", "yellow-80".
     
     
if selected =="Data Donation":
    # st.title(f"Data Donation") 
    nice_header(label="Data Donation", description="This is the data donation page",  color_name="light-blue-70")
    # Donate
    folder='data'
    donateData(folder)
 
     # TODO: 2. Ask the user to also provide metadata for the file....


if selected =="Data Analysis":
    
    # st.title(f"Data Analysis")
    nice_header(label="Data Analysis", description="This is the data analysis page",  color_name="light-blue-70")

    controls_cols=st.columns(2,gap='small')
    with controls_cols[0]:
        from_date = selected_from_dates = st.date_input("From: ", datetime.date(2019, 7, 6)) 
    with controls_cols[1]:
        to_date=selected_to_dates = st.date_input("To: ", datetime.date(2024, 10, 30)) 
    # uploaded_file = st.sidebar.file_uploader("Choose json file", type="json")
    
    statistics = pd.DataFrame()
    
    files_df, selected_files = read_json_files_from_directory(directory="data")
    
 
    
    # st.write("Numbers of files", len(selected_files))
    
    df = pd.DataFrame()
    if (len(selected_files) >=1 ):
        rows = []
        for instant_file in selected_files:
             
            folder='data'
            file_path = os.path.join(folder, instant_file)
            
            f = open(file_path, 'r', encoding='utf-8')
            #data = json.load(uploaded_file)
            #comments = [entry['data'][0]['comment']['comment'] for entry in data['comments_v2']]
            # st.write(file_path)
            data = json.load(f)
            comments_data = data['comments_v2']
            
            #Start
            # comments_data = json.loads(data.decode())['comments_v2']
            
            
            for comment in comments_data:
              current_comment = comment['data'][0]['comment']
            
              # Convert the timestamp to a datetime object
              datetime_object = datetime.datetime.fromtimestamp(current_comment['timestamp'])
            
              # Format the datetime object as a date string (e.g., YYYY-MM-DD)
              date_string = datetime_object.strftime('%Y/%m/%d')
              rows.append({
                  'date':datetime_object,
                  'comment': current_comment['comment']
              })
            
             
            df = pd.DataFrame(rows)
            # print(df)
              
            # Apply the function to each comment in the DataFrame
            # remove_stopwords
            # preprocess
        df['comment'] = df['comment'].apply(remove_stop_words)
                
                # st.write("Dataframe created with shape", df.shape)
                
                # Convert date input to Pandas Timestamp objects for comparison
                # st.write(df.describe())
        from_date = pd.to_datetime(from_date)
        to_date = pd.to_datetime(to_date)
            
        # df.head()
       
        #Filter data according to selected dates
        filtered_df = df[(df['date'] >= from_date) & (df['date'] <= to_date)] 
        
        # Clean comments
        filtered_df["comment"]=filtered_df["comment"].str.lower() #Set commemts to lower case
        filtered_df['comment']=filtered_df['comment'].apply(remove_emojis) #Remove Emojis
        filtered_df['comment']=filtered_df['comment'].apply(remove_malformed_emojis) 
        
        filtered_df['comment']=filtered_df['comment'].apply(remove_non_english)
        
        filtered_df['comment']=filtered_df['comment'].apply(remove_punc) #Remove punctutations
         
        filtered_df['comment']= filtered_df['comment'].apply(tokenize_text) #Tokenize comments
        filtered_df['comment'] = filtered_df['comment'].apply(lambda x: replace_abbreviations(x)) #Replace abbreviations with equivalents
        filtered_df['comment'] = filtered_df['comment'].apply(remove_stop_words)  #Remove stop words
         
        #Compute Subjectivity
        filtered_df['subjectivity'] = filtered_df['comment'].apply(get_subjectivity)
        # Apply sentiment analysis
        # filtered_df['sentiment_score'] = filtered_df['comment'].apply(analyze_sentiment2)
        
    

             
            # Print subjectivity scores
            # print(filtered_df[['comment', 'subjectivity']])
            
        # Compute the combined subjectivity score (average)
        combined_subjectivity_score = filtered_df['subjectivity'].mean()
         
            #End 
             
             
        avg_comment_polarity = analyze_sentiment(filtered_df['comment'])
            # st.write(avg_comment_polarity)
            
        avg_comment_length, max_comment_length = calculate_metrics(filtered_df['comment'])
            # st.write( avg_comment_length, max_comment_length)
             
            
        print("statistics1 ", statistics.shape )
               
        statsdata = {  
            'avg_comment_polarity': [avg_comment_polarity],  
            'avg_comment_length': [avg_comment_length] ,
            'max_comment_length':[max_comment_length]
            }  
        statistics = pd.DataFrame(statsdata ) 
        print("statistics2 ", statistics.shape )
          
        
            # Dashboard Main Panel
        col = st.columns((2, 8), gap='small') 
        stats=st.columns(5,gap='small')  
        with stats[0]: 
            st.info('Polarity',icon="🎭")
            st.metric(label="Polarity",value=f"{avg_comment_polarity*100:,.2f}%")  
        with stats[1]: 
            st.info('Subjectivity',icon="🤥")
            st.metric(label="Subjectivity",value=f"{combined_subjectivity_score*100:,.2f}%")
        with stats[2]:
            st.info('Max. Comm',icon="📈")
            st.metric(label="Max. Comm",value=f"{max_comment_length:,.0f}")
        with stats[3]:
            st.info('Av. Comm',icon="📊")
            st.metric(label="Av. Comm",value=f"{avg_comment_length:,.2f}")
        with stats[4]:
            st.info('Total Comm',icon="💯")
            st.metric(label="Av. Comm",value=f"{filtered_df.shape[0]:,.0f}")
        style_metric_cards(background_color="#FFFFFF",border_left_color="#686664",border_color="#000000",box_shadow="#F71938")
        
         # Process comments for visualizations
        common_words, combined_text = process_comments(filtered_df['comment'])
        
        # st.subheader("Word Cloud")
        generate_word_cloud(combined_text)
        
        # st.subheader("Bar Chart of Common Words")
        generate_bar_chart(common_words) 
        
         
        # Apply sentiment analysis
        filtered_df['sentiment_score'] = filtered_df['comment'].apply(analyze_comment_sentiment)
        
        # st.write( filtered_df.shape)
        # st.write( filtered_df.head())
        # # Group by date and calculate average sentiment
        # avg_sentiment_per_day = filtered_df.groupby('date')['sentiment_score'].mean()
        # st.write(avg_sentiment_per_day)
        # st.write(type(avg_sentiment_per_day))
        # # Plotting
        # fig, ax = plt.subplots()
        # ax.plot(avg_sentiment_per_day[0], avg_sentiment_per_day[1])
        # ax.set_title("Sine Wave")
        # ax.set_xlabel("X-axis")
        # ax.set_ylabel("Y-axis")   
        
        # # Display the plot in Streamlit
        # st.pyplot(fig)
      


        
    
     