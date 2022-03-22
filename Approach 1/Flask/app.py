from re import sub
from gensim.utils import simple_preprocess
import gensim.downloader as api
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from gensim.similarities import WordEmbeddingSimilarityIndex
from gensim.similarities import SparseTermSimilarityMatrix
from gensim.similarities import SoftCosineSimilarity
import numpy as np
import json
import pandas as pd
import pickle

from flask import Flask, render_template, request, url_for
# from utilities.search import Search


app = Flask(__name__)

# import tensorflow as tf
# import pandas as pd
# import nmslib
# from transformers import AlbertTokenizer, TFAlbertModel
# from transformers import  AlbertConfig


# albert_tokenizer = AlbertTokenizer.from_pretrained("albert-base-v2") 


# config = AlbertConfig.from_pretrained('./albert', output_hidden_states=True)

# model = TFAlbertModel.from_pretrained('./albert', config=config,  from_pt=True)

# df = pd.read_csv('final_search.csv')

# search_index = nmslib.init(method='hnsw', space='cosinesimil')


# search_index.loadIndex('./final.nmslib')


# def search(query):
# 	e = albert_tokenizer.encode(query.lower())
# 	input = tf.constant(e)[None, :] 
# 	output = model(input)
# 	v = [0]*768
# 	for i in range(-1, -13, -1):
# 		v = v + output[2][i][0][0] 
# 	emb = v/12
# 	idxs, dists = search_index.knnQuery(emb, k=5)
# 	all_funcs = []
# 	list_of_dist = []
# 	list_of_git = []
# 	for idx, dist in zip(idxs, dists):
# 		if(float(dist)>0.05):
# 			continue
# 		code = df['original_function'][idx]
# 		list_of_dist.append(dist)
# 		list_of_git.append(df['url'][idx])
# 		code = re.sub(r'"""(.*)?"""\s\n',r' ',code,flags=re.DOTALL)
# 		all_funcs.append(code)
# 	return all_funcs,list_of_dist,list_of_git
class Search:
	# From: https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/soft_cosine_tutorial.ipynb
	@staticmethod
	def preprocess(doc):
		stopwords = ['the', 'and', 'are', 'a']
		doc = sub(r'<img[^<>]+(>|$)', " image_token ", doc)
		doc = sub(r'<[^<>]+(>|$)', " ", doc)
		doc = sub(r'\[img_assist[^]]*?\]', " ", doc)
		doc = sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', " url_token ", doc)
		return [token for token in simple_preprocess(doc, min_len=0, max_len=float("inf")) if token not in stopwords]

	@staticmethod
	def function_retreival(query_string,documents_dict):
		indexs = list(documents_dict.keys())
		documents = list(documents_dict.values())
		# Preprocess the documents, including the query string
		corpus = [Search.preprocess(document) for document in documents]
		query = Search.preprocess(query_string)


		# Load the model: this is a big file, can take a while to download and open
		print("Download Started")
		with open('glove.pkl', 'rb') as file:
			glove = pickle.load(file)
		# glove = api.load("glove-wiki-gigaword-50")    
		similarity_index = WordEmbeddingSimilarityIndex(glove)
		print("Download Ended")
		# Build the term dictionary, TF-idf model
		dictionary = Dictionary(corpus+[query])
		tfidf = TfidfModel(dictionary=dictionary)

		# Create the term similarity matrix.  
		similarity_matrix = SparseTermSimilarityMatrix(similarity_index, dictionary, tfidf)


		# Compute Soft Cosine Measure between the query and the documents.
		# From: https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/soft_cosine_tutorial.ipynb
		query_tf = tfidf[dictionary.doc2bow(query)]

		index = SoftCosineSimilarity(
					tfidf[[dictionary.doc2bow(document) for document in corpus]],
					similarity_matrix)

		doc_similarity_scores = index[query_tf]

		# Output the sorted similarity scores and documents
		sorted_indexes = np.argsort(doc_similarity_scores)[::-1]
		results = []
		for idx in sorted_indexes[:5]:
			print(f'{idx} \t {doc_similarity_scores[idx]:0.3f} \t {documents[idx]} \t {indexs[idx]}')
			results.append([doc_similarity_scores[idx], documents[idx], int(indexs[idx])])
		return results






@app.route('/')
def main_page():
    return render_template("main_page.html")

@app.route('/results', methods=['GET'])
def results_page():
	query = request.args.get('query')
	print(">>>>>",query)
	f = open('../translations6.json',)
	data = json.load(f)
	f.close()
	results = Search.function_retreival(query,data)
	print(results)
	funcs = []
	# funcs,dists,gits = subprocess. check_output("python main.py", shell=True)
	dists,gits = [],[]
	df = pd.read_csv("../train_sorted.csv")
	
	# funcs = [df.iloc[results[0][2]]["original_function"]]
	with open("../function_tokens.txt",'r') as f:
		content = f.readlines()
	
	print(content[results[0][2]])
	for k in range(5):
		newdf = df[df['function_tokens'] == str(content[results[k][2]]).strip()]
		print(newdf.shape)
		if newdf.shape[0] > 0:
			funcs.append(newdf.iloc[0,6])
		else:
			funcs.append(df[df['function_tokens'] == str(content[results[k][2]]).strip()]['original_function'])
		dists.append(results[k][0])
	# with open("../Original_function.txt",'r',encoding='utf-8') as f:
	# 	content = f.readlines()
	# funcs = [content[results[0][2]]]
	print(funcs)
	values = len(funcs)
	print(values)
	return render_template("results_page.html",data=query, result = values, codes=funcs,dist=dists,git=gits)

if __name__ == "__main__":
    app.run(debug = True)
