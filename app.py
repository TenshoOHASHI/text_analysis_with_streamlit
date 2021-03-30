
# Core Pkgs 
import streamlit as st
import streamlit.components.v1 as stc 

# NLP Pkgs 
import neattext.functions as nfx 
from wordcloud import WordCloud
from collections import Counter
from textblob import TextBlob
import neattext

# NLP PKgs For ZH
import jieba 
from jieba import analyse


import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# EDA
import pandas as pd

# Data Viz Pkgs 
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt 

import seaborn as sns 
import altair as alt 
plt.style.use("ggplot") 

# Fx 

def plot_wordcloud(docx,lang=None):
    # Font Path and Property 
    if lang == "ZN" or "zn":
        font_path = r"./data/SimHei.ttf"
        font = FontProperties(fname= font_path,size=16)
        mywordcloud = WordCloud(font_path=font_path,background_color="PAPAYAWHIP").generate(docx)

        fig = plt.figure(figsize=(20,10))
        plt.imshow(mywordcloud, interpolation="bilinear")
        plt.axis("off")
        st.pyplot(fig)

    else: 
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

TAGS_ZN ={
    'r'   : 'green',
    'n'   : 'green',
    'nr'  : 'green',
    'nz'  : 'green',
    'PER' : 'green',
    'a'   : 'blue',
    'm'  : 'blue',
    'c'  : 'blue',
    'VBP'  : 'blue',
    'VBZ'  : 'blue',
    'ns'   : 'red',
    'nt'  : 'red',
    'nv'  : 'red',
    't'   : 'cyan',
    'TIME'  : 'cyan',
    'd'  : 'cyan',
    'p'   : 'darkwhite',
    'vn'  : 'darkyellow',
    'ORG' : 'magenta',
    'LOC' : 'magenta',
    'q'   : 'black',
    'u'   : 'black',
    'ad'   : 'black',
    'v'  : 'black',
    'vd'   : 'black',
    'an'  : 'black',
    'xc'  : 'black',
    'f'   : 'yellow',
    'w'   : 'yellow',
    'nw'   : 'yellow',
}

TAGS_ZN_T ={
    'green': 'r',
    'green': 'n',
    'green' :'nr',
    'green': 'nz',
    'green': 'PER',
    'blue':  'a',
    'blue':  'm',
    'blue':  'c',
    'blue':  'VBP',
    'blue':  'VBZ',
    'red':   'ns',
    'red':   'nt',
    'red':   'nv',
    'cyan':  't',
    'cyan':  'TIME',
    'cyan':  'd',
    'darkwhite': 'p',
    'darkyellow': 'vn',
    'magenta': 'ORG',
    'magenta': 'LOC',
    'black':  'q',
    'black':  'u',
    'black':  'ad',
    'black':  'v',
    'black':  'vd',
    'black':  'an',
    'black':  'xc',
    'yellow': 'f',
    'yellow': 'w',
    'yellow': 'nw',
}


def mytag_visualizer_zn(tagge_docx):
    
    colored_text = []
    token_ = []
    color_tag_ = []
    tag_color_ = []
    for token, tag in tagge_docx.items():
        if tag in TAGS_ZN.keys():
            toeken = token
            color_for_tag = TAGS_ZN.get(tag)
            tag_for_color = TAGS_ZN_T.get(color_for_tag)
            result =  '<span style="color:{}">{}</span>'.format(color_for_tag,token)
            colored_text.append(result)
            token_.append(token)
            color_tag_.append(color_for_tag)
            tag_color_.append(tag_for_color)

    token_color_df = pd.DataFrame([token_, color_tag_,tag_color_]).T
    token_color_df.columns = ["tokens","colors","tags"]

    st.write(token_color_df)
    result = "".join(colored_text)
    return result

def mytag_visualizer(tagge_docx):
    colored_text = []
    for i in tagge_docx:
        if i[1] in TAGS.keys():
            token = i[0]
            color_for_tag = TAGS.get(i[1])
            result = '<span style="color:{}">{}</span>'.format(color_for_tag,token)
            colored_text.append(result)
    result = " ".join(colored_text)
    
    return result

# Split word into to list 
def preprocess_tokens_list(text):
    # Load Stopwords 
    with open("./stopwords/stopwords.txt","r") as f:
        stopwords_zn = [word.strip() for word in f ] 

    seg_list = jieba.cut(text,cut_all=True,) # return list 
    preprossed_tokens_list = " ".join([token for token in seg_list if token not in stopwords_zn])
    
    return preprossed_tokens_list

def plot_mendelhall_curve_zn(docx):
    raw_test_token_list = jieba.cut(docx,cut_all=True)
    raw_test_token_str = " ".join(raw_test_token_list)
    # Call fx 
    plot_mendelhall_curve(raw_test_token_str)

def most_common_keyword_extracter_idf_zn(docx,common=10):
    coommon_keyword_idf_list = analyse.extract_tags(docx,topK=common,withWeight=True)
    coommon_keyword_idf_df = pd.DataFrame(coommon_keyword_idf_list,columns=["Word Most Common","Inverse Document Frequency"])
    
    st.dataframe(coommon_keyword_idf_df)

def pos_jieba_zn(raw_text):
    pseg = jieba.posseg.cut(raw_text)

    words_tags_list = []

    for i in pseg:
        words_tag_str = (i.__str__().decode("utf-8"))
        words_tags_list_ = (words_tag_str.split("/"))
    
    
        #words_tags_list.extend(words_tags_list_)
        for w_t in (words_tags_list_):
            a = "".join(list(w_t))
            words_tags_list.append(a)
    
    df = pd.DataFrame(words_tags_list,columns=["pos"])
    words = df[df["pos"].index % 2  == 0 ].values.tolist()
    tags = df[df["pos"].index % 2 == 1 ].values.tolist()

    words_tags_dict = {}
    for w,t  in zip(words,tags):
       
        words_tags_dict[w[0]] = t[0]

    return words_tags_dict 



def main():
    st.title("Text Analysis App")

    menu = ["English","Chinese","Japanese"]
    Choice = st.sidebar.selectbox("Menu",menu)

    if Choice == "English":
        st.subheader("EN")
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
                
    elif Choice == "Chinese":
        st.subheader("中文")

        # Text Area 
        raw_text = st.text_area("请输入文本")
    
        # Preprocess
        processed_text = preprocess_tokens_list(raw_text)

        #pseg_list_tup = {words:tags for words, tags in pseg }
        #st.write((words_tag_dic))
       
        # submit bottom 
        if st.button("Submit"):
            if len(raw_text) > 2:
                st.success("Processing")
            elif len(raw_text) == 1:
                st.warning("Insufficient Text,Minimum is 2") 
            else:
                st.warning("请输入文本") 
         # Basic tf-idf
        
        if len(processed_text) > 1 :
            st.subheader("逆向文件频率")
            most_common_keyword_extracter_idf_zn(processed_text)
            st.text("——————————————————"*5)
    
        # Layout 
        col1,col2 = st.beta_columns(2)
        with col1:
            with st.beta_expander("原文本"):
                st.write(raw_text)
        with col1:
            with st.beta_expander("划分词类"):
                if len(raw_text) > 1: 
                    st.success("Part of Speech")
                    words_pos_tagged_dict = pos_jieba_zn(raw_text)
                    
                    #st.write(words_pos_tagged_dict)
                    tagged_span_color_html = mytag_visualizer_zn(words_pos_tagged_dict)
                    stc.html(tagged_span_color_html,scrolling=True)
                else: st.warning("请输入文本")

        with col1:
            with st.beta_expander("词出现频率展示图"):
                max_limit = len(processed_text.split())
                if len(processed_text) > 1:
                    try:
                        num_of_tokens = st.number_input("num of tokens",10,max_value=max_limit)
                        plot_word_freq(processed_text,num_of_tokens)
                    except:st.info("请输入足够的文本")
                else: st.warning("请输入文本")

        with col2:
            with st.beta_expander("处理后的文本"):
                if len(processed_text) >1:
                    st.write(processed_text)
                else:st.warning("已输入的文本皆焉停用词，请再输入其他文本")   

        with col2:
            with st.beta_expander("展示词云"):
                if len(processed_text) >1 :
                    st.success("Word Plot")
                    plot_wordcloud(processed_text,lang="zn")
                else: st.warning("请输入文本")
                

        with col2:
            with st.beta_expander("词长度和出现频率展示图"):
                if len(raw_text) > 1:
                    st.success("Mendehall Curve")
                    plot_mendelhall_curve_zn(raw_text)

                else: st.warning("请输入文字")
            

                
    else:
        st.subheader("JA")


if __name__ == "__main__":
    main()