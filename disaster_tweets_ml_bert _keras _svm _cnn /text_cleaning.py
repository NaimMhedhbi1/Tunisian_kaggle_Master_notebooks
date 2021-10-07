# %% [code]
import pandas as pd
import nltk 
from bs4 import BeautifulSoup 
import string 
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer 
from nltk.stem import WordNetLemmatizer 
from nltk.stem.porter import PorterStemmer 
df = pd.read_csv('../input')
""" As a data scientist, 
we may use NLP for sentiment 
analysis (classifying words to
 have positive or negative connotation) 
or to make predictions in classification models,
 among other things. Typically, whether we’re given 
 the data or have to scrape it, the text will be in 
 its natural human format of sentences, paragraphs,
 tweets, etc. From there, before we can dig into analyzing,
 we will have to do some cleaning to break the text down into 
 a format the computer can easily understand."""


print (df.shape)
print(df['customer_reviews'][3]) 

#sepearte review column by // breaks 
reviews = df['customer_reviews'].str.split("//",n = 4, expand = True) 
print(reviews.head()) 

#Then you can rename the new 0, 1, 2, 3, 4 columns in the original reviews_df and drop the original messy column.

df['review_tittle'] = reviews[0] 
df['rating'] = reviews[1] 
df['review_date'] = reviews[2] 
df['customer_name'] = reviews[3] 
df['review'] = reviews[4]  
df.drop(columns = 'customer_reviews' , inplace = True) 


#removing HTML 
def remove_html(text) : 
    soup = BeautifulSoup(text,'lxml')
    html_free = soup.get_text()
    return html_free 
#df['review'] = df['review'].apply(lambda x:remove_html(x))  

#df['review_count'] = df['review'].str.split().apply(len).value_counts()
 



df['review'] =  df['review'].astype(str)
df['reviewword_count'] = df['review'].str.split().map(len)
df['charreview_count'] = df['review'].str.len()



"""Generally, while solving an NLP problem, 
the first thing we do is to remove the stopwords. 
But sometimes calculating the number of stopwords 
can also give us some extra information which we might
 have been losing before. """ 
 
import nltk

from nltk.corpus import stopwords
stop = stopwords.words('english')
df['stopwords'] = df['review'].apply(lambda x: len([x for x in x.split() if x in stop]))

df1 = df[['review','reviewword_count','charreview_count','stopwords']]

df['review'] = df['review'].str.replace('[^\w\s]','') 


from textblob import TextBlob
df['review'].apply(lambda x: str(TextBlob(x).correct()))

  