
# Core Pkgs 
import streamlit as st
import streamlit.components.v1 as stc 

# NLP Pkgs 
import neattext.functions as nfx 
from wordcloud import WordCloud
from collections import Counter
from textblob import TextBlob
import neattext

import nltk
nltk.download('punkt')

# EDA
import pandas as pd

# Data Viz Pkgs 
import matplotlib.pyplot as plt 
import seaborn as sns 
import altair as alt 
plt.style.use("ggplot") 

# Fx 

def plot_wordcloud(docx):
    mywordcloud = WordCloud().generate(docx)
    fig = plt.figure(figsize=(20,10))
    plt.imshow(mywordcloud, interpolation="bilinear")
    plt.axis("off")
    st.pyplot(fig)

def plot_word_freq(docx,num=10):
    word_freq_list = Counter(docx.split())
    most_common_tokens = word_freq_list.most_common(num)
    # DataFrame 
    most_common_tokens = pd.DataFrame(data=most_common_tokens, columns=["Words","Counts"])
    
    fig = plt.figure()
    sns.barplot(x="Words",y="Counts", data=most_common_tokens)
    plt.title("Word Counts")
    plt.xticks(rotation=45)
    plt.show()
    st.pyplot(fig)

def plot_word_freq_with_altair(docx,num=10):
    word_freq_list = Counter(docx.split())
    most_common_tokens = dict(word_freq_list.most_common(num))
    word_freq_df = pd.DataFrame({"Tokens":most_common_tokens.keys(),"Counts":most_common_tokens.values()})
    
    brush = alt.selection(type="interval",encodings=["x"])
    c = alt.Chart(word_freq_df).make_bar().encode(
        x = "Tokens",
        y = "Counts",
        opacity = alt.condition(brush, alt.OpacityValue(1), alt.OpacityValue(0.7)),
    ).add_selection(brush)

    st.altair_chart(c, use_container_width=True)

def plot_mendelhall_curve(docx):
    word_length = [len(token) for token in docx.split()]
    word_length_count = Counter(word_length)
    sorted_word_length_count = sorted(dict(word_length_count).items())

    x,y = zip(*sorted_word_length_count)
    mendelhall_df = pd.DataFrame({"Tokens":x,"Counts":y})
    st.line_chart(mendelhall_df["Counts"],use_container_width=True)



def get_plot_pos_tags(docx):
    blob = TextBlob(docx)
    tagge_docx = blob.tags
    tagged_df = pd.DataFrame(tagge_docx,columns=["token","tags"])

    #token_count = dict(Counter(tagged_df["token"].tolist())) 
    return tagged_df 

TAGS =  {
    'NN'   : 'green',
    'NNS'  : 'green',
    'NNP'  : 'green',
    'NNPS' : 'green',
    'VB'   : 'blue',
    'VBD'  : 'blue',
    'VBG'  : 'blue',
    'VBN'  : 'blue',
    'VBP'  : 'blue',
    'VBZ'  : 'blue',
    'JJ'   : 'red',
    'JJR'  : 'red',
    'JJS'  : 'red',
    'RB'   : 'cyan',
    'RBR'  : 'cyan',
    'RBS'  : 'cyan',
    'IN'   : 'darkwhite',
    'POS'  : 'darkyellow',
    'PRP$' : 'magenta',
    'PRP$' : 'magenta',
    'DET'   : 'black',
    'CC'   : 'black',
    'CD'   : 'black',
    'WDT'  : 'black',
    'WP'   : 'black',
    'WP$'  : 'black',
    'WRB'  : 'black',
    'EX'   : 'yellow',
    'FW'   : 'yellow',
    'LS'   : 'yellow',
    'MD'   : 'yellow',
    'PDT'  : 'yellow',
    'RP'   : 'yellow',
    'SYM'  : 'yellow',
    'TO'   : 'yellow',
    'None' : 'off'
}


def mytag_visualizer(tagge_docx):
    colored_text = []
    for i in tagge_docx:
        if i[1] in TAGS.keys():
            token = i[0]
            color_for_tag = TAGS.get(i[1])
            result = '<span style="color:{}">{}"</span>'.format(color_for_tag,token)
            colored_text.append(result)
    result = " ".join(colored_text)
    #print(result)
    return result




def main():
    st.title("Text Analysis App")

    menu = ["Home","About"]
    Choice = st.sidebar.selectbox("Menu",menu)

    if Choice == "Home":
        st.subheader("Home")
        # Text Area 
        raw_text = st.text_area("Enter Text Here")
        if st.button("Submit"):
            if len(raw_text) > 2:
                st.success("Processing")
            elif len(raw_text) == 1:
                st.warning("Insufficient Text,Minimum is 2") 
            else:
                st.write("Enter Text") 
        # if raw_text is not None:
        #    st.sucess(raw_text)

        # Layout 
        col1,col2 = st.beta_columns(2)
        processed_text = nfx.remove_stopwords(raw_text)
        processed_text_deep = nfx.remove_special_characters(processed_text)

        with col1:
            with st.beta_expander("Original Text"):
                st.write(raw_text)
                
            with st.beta_expander("Pos Tagger Text"):
                #tagged_docx = get_plot_pos_tags(processed_text)
                #st.dataframe(tagged_docx)
                st.success("Part Of Speech")
                # Components HTML
                tagged_docx = TextBlob(raw_text).tags # generate tags 
                
                tagged_span_color_html = mytag_visualizer(tagged_docx) 
                stc.html(tagged_span_color_html,scrolling=True)
            try:
                with st.beta_expander("Plot Word Freqency"):
                    max_limit = len(processed_text.split())
                    num_of_tokens = st.number_input("Num of tokens",10,max_limit)
                    plot_word_freq(processed_text,num_of_tokens)
                    #plot_word_freq_with_altair(processed_text,num_of_tokens)
            except:
                pass


        with col2:
            
                with st.beta_expander("Processed Text"):
                    processed_text = nfx.remove_stopwords(raw_text)
                    st.write(processed_text_deep)
                try:    
                    with st.beta_expander("Plot WordCloud"):
                        st.success("Wordcloud")
                        plot_wordcloud(processed_text_deep)
                except:
                    pass
                    
                with st.beta_expander("Plot Stylpmetry Curve"):
                    st.success("Mendehall Curve")
                    if len(raw_text) > 2:   
                        plot_mendelhall_curve(raw_text)  
                
    else:
        st.subheader("About")



if __name__ == "__main__":
    main()