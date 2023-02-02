#add your code here
import collections
import json
import pandas as pd
import argparse
import re
import string
from ast import literal_eval
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
                index = faiss,
                use_cuda = False):
        
    self.model = SentenceTransformer(retriever_model)
    self.index = index
    self.embedding_size = embedding_size
    self.device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
    self.retriever_type = retriever_type

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
                reader_model="mrm8488/bert-mini-finetuned-squadv2", 
                use_cuda = False,):
        
    self.model_name = reader_model
    self.device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
    self.pipe = None

  def __call__(self, file_name ="model_quantized.onnx", save_directory= "tmp/onnx/", truncation = "only_second", stride = 128, n_best_size=20):
    
    self.quantize_model()

    reader_model = ORTModelForQuestionAnswering.from_pretrained(save_directory, file_name="model_quantized.onnx")
    tokenizer = AutoTokenizer.from_pretrained(save_directory)

    self.pipe = pipeline("question-answering",
                    model=reader_model,
                    tokenizer=tokenizer,
                    truncation= truncation,
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

  
  def read(self, question, passages, corpus_ids, distances):
    # We extract corpus ids and scores for the each query
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
      pred_out.append(ans)
    else:
      pred_out.append(outputs[0]["answer"])

    return pred_out
  
  # add loading of meta model





def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reader_model", type=str, default="vaibhav9/distil-roberta-qa")
    parser.add_argument("--retriever_model", type=str, default="vaibhav9/distil-roberta-qa")
    parser.add_argument("--input_dir", type=str, default="datasets/train_data.csv")
    parser.add_argument("--output_pred", type=str, default="oodomain_train/")
    parser.add_argument("--use_cuda", action="store_false")
    args = parser.parse_args()
    
    df = pd.read_csv(args.input_dir)
    retriever = Retriever()
    index = retriever(file_name)
    # qg_model = AutoModelForSeq2SeqLM.from_pretrained("valhalla/t5-small-qg-hl")
    # qg_tokenizer = AutoTokenizer.from_pretrained("valhalla/t5-small-qg-hl")
    # qg_pipe = CustomPipeline(model = qg_model, 
    #                          tokenizer=qg_tokenizer, 
    #                          use_cuda=args.use_cuda,
    #                         qa_checkpoint = args.qa_model)
    
    # all_themes = df["theme"].unique()
    # theme_dict = {}
    # for theme in all_themes:
    #     theme_dict[theme] = qg_pipe.get_theme_dataset(df,theme).drop_duplicates()
    # out = pd.concat(list(theme_dict.values()))
    # qg_pipe.save_to_json(out, args.output_dir)
    #out.to_csv(args.output_dir + "synthetic_data.csv")
    

if __name__ == "__main__":
    main()