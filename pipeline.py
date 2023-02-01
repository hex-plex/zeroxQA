#add your code here
import collections
import json
import pandas as pd
import re
import string
import timeit
from ast import literal_eval
import time
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
                retriever_type = "single",
                use_cuda = False,):
        
    self.model = SentenceTransformer(retriever_model)
    self.index = faiss 
    self.embedding_size = embedding_size
    self.device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
    self.reader_type = retriever_type

  def __call__(self, corpus, n_clusters = 4, n_probe = 3):
    corpus_json = json.loads(pd.read_csv(corpus).to_json(orient="records"))
    passages = []
    for row in corpus_json :
      passages.append(row['paragraph'])

    # Setup Faiss

    #We use Inner Product (dot-product) as Index. We will normalize our vectors to unit length, then is Inner Product equal to cosine similarity
    quantizer = self.index.IndexFlatIP(self.embedding_size)
    index = self.index.IndexIVFFlat(quantizer, self.embedding_size, n_clusters, faiss.METRIC_INNER_PRODUCT)
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

      return index

  def encode(self,question):
    question_embeddings = self.model.encode(question)
    question_embeddings = question_embeddings / np.linalg.norm(question_embeddings)
    question_embeddings = np.expand_dims(question_embeddings, axis=0)
    
    return question_embeddings
  
  def search(self,question_embedding, top_k):
    distances, corpus_ids = self.index.search(question_embedding, top_k)
    return distances, corpus_ids

class Reader:
  def __init__(self, 
                reader_model="mrm8488/bert-mini-finetuned-squadv2", 
                use_cuda = "cpu",):
        
    self.model_name = reader_model
    self.device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"

  def __call__(self, file_name ="model_quantized.onnx", save_directory= "tmp/onnx/", truncation = "only_second", stride = 128, n_best_size=20):
    
    self.export_to_onnx()

    reader_model = ORTModelForQuestionAnswering.from_pretrained(save_directory, file_name="model_quantized.onnx")
    tokenizer = AutoTokenizer.from_pretrained(save_directory)

    pipe = pipeline("question-answering",
                    model=reader_model,
                    tokenizer=tokenizer,
                    truncation= truncation,
                    stride=stride,
                    padding="max_length",
                    n_best_size = n_best_size)
    
    return pipe
  
  def export_to_onnx(self, save_directory= "tmp/onnx/"):

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

  
  def read(self, question, passages, corpus_ids, distances):
    # We extract corpus ids and scores for the first query
    hits = [{'corpus_id': id, 'score': score} for id, score in zip(corpus_ids[0], distances[0])]
    hits = sorted(hits, key=lambda x: x['score'], reverse=True)

    ans = {}
    ans["paragraph_id"]=-1
    ans["answers"]=""
    ans["question_id"] = question["id"]
    outputs=[]

    pred_out = []

    for hit in hits :
      # print(hit["corpus_id"])
      if hit['corpus_id'] != -1:
        # print("inside")
        context=passages[hit['corpus_id']]["paragraph"]
        output = pipe(question=question["question"], context=context, handle_impossible_answer= True)
    
        if output["score"] > 0.5 and output["answer"]:
          outputs.append({
            "score" : output["score"],
            "answer" : {
              "question_id" : question["id"],
              "paragraph_id" : hit["corpus_id"]+1,
              "answers" : output["answer"]
            } 
          })

    outputs = sorted(outputs, key=lambda x: x['score'], reverse=True)

    if not outputs:
      # print("inside")
      pred_out.append(ans)
    else:
      # print(outputs)
      pred_out.append(outputs[0]["answer"])

    return pred_out


    


