import collections
import json
import pandas as pd
import argparse
import re
import string
import gzip
import os
import torch
import csv
import pickle
import faiss
import numpy as np
from transformers import pipeline
from optimum.onnxruntime import ORTModelForQuestionAnswering
from transformers import AutoTokenizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from optimum.onnxruntime import ORTQuantizer
from sentence_transformers import SentenceTransformer



class Retriever:
  def __init__(self, 
                retriever_model="sentence-transformers/all-MiniLM-L12-v2", 
                embedding_size = 384,
                use_cuda = False,
                retriever_type = "single",
                indexing = faiss):
        
    self.model = SentenceTransformer(retriever_model)
    self.index_type = indexing
    self.embedding_size = embedding_size
    self.device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
    self.retriever_type = retriever_type
    self.index = None

  def __call__(self, corpus, n_clusters = 4, n_probe = 3):
    corpus_json = json.loads(pd.read_csv(corpus).to_json(orient="records"))
    passages = []
    for row in corpus_json :
      passages.append(row['paragraph'])
    
    # Setup Faiss
    length_passages = len(passages)
    n_clusters = min(length_passages,int(8*length_passages**0.5))
    n_probe = min(3,length_passages)

    #We use Inner Product (dot-product) as Index. We will normalize our vectors to unit length, then is Inner Product equal to cosine similarity
    quantizer = self.index_type.IndexFlatIP(self.embedding_size)
    index = self.index_type.IndexIVFFlat(quantizer, self.embedding_size, n_clusters, faiss.METRIC_INNER_PRODUCT)
    index.nprobe = n_probe
    
    if self.retriever_type is "single":
      
      corpus_embeddings = self.model.encode(passages, convert_to_numpy=True, show_progress_bar=True)

      ### Create the FAISS index
      print("Start creating FAISS index")
      # First, we need to normalize vectors to unit length
      corpus_embeddings = corpus_embeddings/ np.linalg.norm(corpus_embeddings, axis=1)[:, None]

      # # Then we train the index to find a suitable clustering
      index.train(corpus_embeddings)

      # # Finally we add all embeddings to the index
      index.add(corpus_embeddings)

      self.index = index

      return index

  def question_encode(self,question):
    question_embeddings = self.model.encode(question)
    question_embeddings = question_embeddings / np.linalg.norm(question_embeddings)
    question_embeddings = np.expand_dims(question_embeddings, axis=0)
    
    return question_embeddings
  
  def search(self,question_embedding, top_k = 5):
    distances, corpus_ids = self.index.search(question_embedding, top_k)
    return distances, corpus_ids

class Reader:
  def __init__(self,
                reader_model="mrm8488/bert-mini-5-finetuned-squadv2",
                theme = None,
                theme_dict = None,
                use_cuda = False,):
    self.theme = theme
    if self.theme is None:
      self.model_name = reader_model
    else:
      file = open(theme_dict, 'rb')
      # dump information to that file
      data = pickle.load(file)
      # close the file
      file.close()
      self.model_name = data[theme]
    self.device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
    self.pipe = None

  def __call__(self, stride = 128, n_best_size=20, file_name = "model_quantized.onnx", save_directory= "tmp/onnx/"):
    
    self.quantize_model()

    reader_model = ORTModelForQuestionAnswering.from_pretrained(save_directory, file_name=file_name, from_transformers=True) #from from_transformers=True
    tokenizer = AutoTokenizer.from_pretrained(save_directory)

    self.pipe = pipeline("question-answering",
                    model=reader_model,
                    tokenizer=tokenizer,
                    truncation= "only_second",
                    stride=stride,
                    padding="max_length",
                    n_best_size = n_best_size)
    
    return self.pipe
  
  def quantize_model(self, save_directory= "tmp/onnx/"):

    # Load a model from transformers and export it to ONNX
    ort_model = ORTModelForQuestionAnswering.from_pretrained(self.model_name, from_transformers=True)
    tokenizer = AutoTokenizer.from_pretrained(self.model_name)
    # Save the onnx model and tokenizer
    ort_model.save_pretrained(save_directory)
    tokenizer.save_pretrained(save_directory)
    qconfig = AutoQuantizationConfig.arm64(is_static=False, per_channel=False)
    quantizer = ORTQuantizer.from_pretrained(ort_model)
    # Apply dynamic quantization on the model
    quantizer.quantize(save_dir=save_directory, quantization_config=qconfig)

  
  def read(self, question, passages, corpus_ids, distances, top_k_hits=3):
    # We extract corpus ids and scores for the each query
    hits = [{'corpus_id': id, 'score': score} for id, score in zip(corpus_ids[0], distances[0])]
    hits = sorted(hits, key=lambda x: x['score'], reverse=True)

    ans = {}
    ans["paragraph_id"]=-1
    ans["answers"]=""
    ans["question_id"] = question["id"]
    outputs=[]

    pred_out = []

    for hit in hits[0:top_k_hits] :
      # print(hit["corpus_id"])
      if hit['corpus_id'] != -1:
        # print("inside")
        context=passages[hit['corpus_id']]
        output = self.pipe(question=question["question"], context=context, handle_impossible_answer= True)
    
        if output["score"] > 0.5 and output["answer"]:
          outputs.append({
            "score" : output["score"],
            "answer" : {
              "question_id" : question["id"],
              "paragraph_id" : hit["corpus_id"]+1,
              "answers" : output["answer"]
            } 
          })

    outputs = sorted(outputs, key=lambda x: -x['score'])

    if not outputs:
      return ans
    else:
      return outputs[0]["answer"]
  
  # add loading of meta model