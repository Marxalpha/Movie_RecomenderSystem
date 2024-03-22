import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
from ast import literal_eval

movies_df=pd.read_csv('./input/movies_metadata.csv',low_memory=False)
credits_df=pd.read_csv('./input/credits.csv')
ratings_df=pd.read_csv('./input/ratings.csv')

for index,row in movies_df.iterrows():
	try:
		row["id"]=int(row["id"])
	except:
		movies_df.drop(index,axis=0,inplace=True)
credits_df["id"]=credits_df["id"].astype(object)
movies_df["id"]=pd.to_numeric(movies_df["id"])
movies_copy0=pd.merge(movies_df,credits_df,how='left',right_on='id',left_on='id')

c=movies_copy0["vote_average"].mean()
m=movies_copy0["vote_count"].quantile(0.9)
movie_list=movies_copy0.copy().loc[movies_copy0['vote_count']>=m]

def rating(x,M=m,C=c):
    v=x['vote_count']
    r=x['vote_average']
    return (r*v + c*m)/(v+m)

movie_list['score']=movie_list.apply(rating,axis=1)

movies_copy0.columns
movie_list=movie_list.sort_values('score',ascending=False)
# moviess=movie_list['title']
# idss=movie_list['id']
tfdif=TfidfVectorizer(stop_words="english")
movies_copy0['overview']=movies_copy0['overview'].fillna('')
overview_arr=movies_copy0['overview'].values

tfidf_matrix=tfdif.fit_transform(overview_arr)
tfidf_matrix_sim=linear_kernel(tfidf_matrix,tfidf_matrix)


indexcs=pd.Series(movies_copy0.index,index=movies_copy0['title']).drop_duplicates()

def getrec_overview(target_movie:str):
    movie_titles=movies_copy0['title'].values
    if target_movie not in movie_titles:
        print(target_movie,"Movie not Recognized")
        return
    
    sims=tfidf_matrix_sim
    idx=indexcs[target_movie]
    index_of_target_movie=np.where(movie_titles==target_movie)[0]
    simsc=list(enumerate(sims[idx]))
    
    top10=sorted(simsc,key=lambda x:x[1],reverse=True)[1:11]
    print("Movies similar to ",target_movie," based on ")
    movieidx=[i[0] for i in top10]
    return movies_copy0.iloc[movieidx]

movies_copy=movies_copy0.copy()
movies_copy['Score']=movies_copy.apply(rating,axis=1)
movies_copy=movies_copy.sort_values('Score',ascending=False)
keywords_df=pd.read_csv('./input/keywords.csv')
# for index,row in movies_copy.iterrows():
# 	try:
# 		row["id"]=int(row["id"])
# 	except:
# 		movies_df.drop(index,axis=0,inplace=True)
# credits_df["id"]=credits_df["id"].astype(object)
# movies_df["id"]=pd.to_numeric(movies_df["id"])
movies_copy=pd.merge(movies_copy,keywords_df,how='left',right_on='id',left_on='id')

movies_copy['crew']=movies_copy['crew'].fillna('')
movies_copy['cast']=movies_copy['cast'].fillna('')
movies_copy['keywords']=movies_copy['keywords'].fillna('')
movies_copy['genres']=movies_copy['genres'].fillna('')

def get_val(row):
    dic=literal_eval(row)   
    return [d['name'] for d in dic]

def get_dir(row):
    dic=literal_eval(row)
    for i in dic:
        if i['job']=='Director':
            return i['name']
for index,row in movies_copy.iterrows():
    if (row['crew']==''):
        movies_copy.drop(index,axis=0,inplace=True)


movies_copy['Director']=movies_copy['crew'].apply(get_dir)
movies_copy['cast']=movies_copy['cast'].apply(get_val)
movies_copy['crew']=movies_copy['crew'].apply(get_val)
movies_copy['genres']=movies_copy['genres'].apply(get_val)
movies_copy['keywords']=movies_copy['keywords'].apply(get_val)

def clean(x):
    if isinstance(x,list):
        return [str.lower(i.replace(" ",""))for i in x]
    else:
        if isinstance(x,str):
            return str.lower(x.replace(" ",""))
        else:
            return ''

features=['cast','crew','genres','keywords','Director']
for i in features:
    movies_copy[i]=movies_copy[i].apply(clean)

def create_fet(x):
    return ' '.join(x['keywords'])+' '+' '.join(x['cast'])+' '+' '.join(x['Director'])+' '+' '.join(x['genres'])

movies_copy['Meta']=movies_copy.apply(create_fet,axis=1)

count=CountVectorizer(stop_words='english')
count_matrix=count.fit_transform(movies_copy['Meta'])
cosinesim=cosine_similarity(count_matrix,count_matrix)
indeces=pd.Series(movies_copy.index,index=movies_copy['title']).drop_duplicates()

def getrec_gdcc(title,cosine_sim=cosinesim):
    idx=indeces[title]
    simsc=list(enumerate(cosine_sim[idx]))
    simsc=sorted(simsc,key=lambda x:x[1],reverse=True)
    simsc=simsc[1:11]
    movie_indices=[i[0] for i in simsc]
    
    return movies_copy.iloc[movie_indices]

def recomend(title):
	x=getrec_overview(title)
	y=getrec_gdcc(title)
	z=pd.concat([x,y])
	z=z.sort_values('Score',ascending=False)
	z=z.drop_duplicates(subset='title')
	print("No of rec" ,z.shape())
	return z['title']

recco=recomend('Avengers: Age of Ultron')
print(recco)