import pandas as pd
import numpy as np
import random, itertools, time, json
from typing import Optional, Dict, Union
from tqdm import tqdm
import torch, transformers,spacy 
from nltk import sent_tokenize
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, AutoModelForSeq2SeqLM
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class CustomPipeline:
    def __init__(self, 
                 model, 
                 tokenizer, 
                 use_cuda,
                 qa_checkpoint,
                 ner_limit = 0,
                 qg_format="highlight", 
                 unans_filter="tfidf"):
        
        self.model = model
        self.tokenizer = tokenizer
        self.qg_format = qg_format
        self.device = "cuda" if (use_cuda and torch.cuda.is_available()) else "cpu"
        self.model.to(self.device)
        self.ner = spacy.load("en_core_web_sm")
        self.qa_pipe = transformers.pipeline("question-answering",
                                    model=qa_checkpoint,
                                    tokenizer=qa_checkpoint)
        self.model_type = "t5"
        self.unans_filter = unans_filter
        self.ner_limit = np.inf if (ner_limit == 0) else ner_limit

    def __call__(self, inputs):
        inputs = " ".join(inputs.split())
        sents, answers = self._extract_answers(inputs)
        flat_answers = list(itertools.chain(*answers))
        
        if len(flat_answers) == 0:
          return []
        if self.qg_format == "prepend":
            qg_examples = self._prepare_inputs_for_qg_from_answers_prepend(inputs, answers)
        else:
            qg_examples = self._prepare_inputs_for_qg_from_answers_hl(sents, answers)
        
        qg_inputs = [example['source_text'] for example in qg_examples]
        questions = self._generate_questions(qg_inputs)
        output = [{'answer': example['answer'], 'question': que} for example, que in zip(qg_examples, questions)]
        return output
    
    def _generate_questions(self, inputs):
        inputs = self._tokenize(inputs, padding=True, truncation=True)
        outs = self.model.generate(
            input_ids=inputs['input_ids'].to(self.device), 
            attention_mask=inputs['attention_mask'].to(self.device), 
            max_length=32,
            num_beams=4,
        )
        
        questions = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in outs]
        return questions
    
    def _extract_answers(self, context):
        sents = sent_tokenize(context)
        answers = []
        start_chars = []
        for sent in sents:
          ents = self.ner(sent).ents
          answers.append([ent.text for cnt, ent in enumerate(ents) if cnt < self.ner_limit])
          #if len(ents) > 0:
          #  answers.append([ents[0].text])
          #else:
          #  answers.append([])
        return sents, answers
    
    def _tokenize(self,
        inputs,
        padding=True,
        truncation=True,
        add_special_tokens=True,
        max_length=512
    ):
        inputs = self.tokenizer.batch_encode_plus(
            inputs, 
            max_length=max_length,
            add_special_tokens=add_special_tokens,
            truncation=truncation,
            padding="max_length" if padding else False,
            pad_to_max_length=padding,
            return_tensors="pt"
        )
        return inputs
    
    def _prepare_inputs_for_ans_extraction(self, text):
        sents = sent_tokenize(text)

        inputs = []
        for i in range(len(sents)):
            source_text = "extract answers:"
            for j, sent in enumerate(sents):
                if i == j:
                    sent = "<hl> %s <hl>" % sent
                source_text = "%s %s" % (source_text, sent)
                source_text = source_text.strip()
            
            if self.model_type == "t5":
                source_text = source_text + " </s>"
            inputs.append(source_text)

        return sents, inputs
    
    def _prepare_inputs_for_qg_from_answers_hl(self, sents, answers):
        inputs = []
        for i, answer in enumerate(answers):
            if len(answer) == 0: continue
            for answer_text in answer:
                sent = sents[i]
                sents_copy = sents[:]
                
                answer_text = answer_text.strip()
                
                ans_start_idx = sent.index(answer_text)
                
                sent = f"{sent[:ans_start_idx]} <hl> {answer_text} <hl> {sent[ans_start_idx + len(answer_text): ]}"
                sents_copy[i] = sent
                
                source_text = " ".join(sents_copy)
                source_text = f"generate question: {source_text}" 
                if self.model_type == "t5":
                    source_text = source_text + " </s>"
                
                inputs.append({"answer": answer_text, "source_text": source_text})
        
        return inputs
    
    def _prepare_inputs_for_qg_from_answers_prepend(self, context, answers):
        flat_answers = list(itertools.chain(*answers))
        examples = []
        for answer in flat_answers:
            source_text = f"answer: {answer} context: {context}"
            if self.model_type == "t5":
                source_text = source_text + " </s>"
            
            examples.append({"answer": answer, "source_text": source_text})
        return examples
    
    @staticmethod
    def get_final_ans(ans1, ans2):
      if len(ans1) < len(ans2):
        return ans2
      else:
        return ans1

    def generate_qa(self, txt):
        final_qg  =[]
        qg = self.__call__(txt)
        outputs = self.qa_pipe(question=[t["question"] for t in qg], context = [txt for i in range(len(qg))])
        if len(qg) == 1:
            outputs = [outputs]
        for idx, output in enumerate(outputs):
            if (output["score"] < 0.7) or ((output["answer"] not in qg[idx]["answer"]) and (qg[idx]["answer"] not in output["answer"])):
              pass
            else:
              final_qg.append({"question":qg[idx]["question"], 
                               "answer":self.get_final_ans(qg[idx]["answer"], output["answer"]), 
                               "start_char":output["start"]})
        return final_qg
    

    def add_unans_que(self, theme_data):
      n = len(theme_data)
      que_to_add = {}
      for i in tqdm(range(n)):
        que_to_add[i] = []
        unans_questions = []
        pop = list(range(n))
        pop.remove(i)
        m = min(n-1, len(theme_data[i]["qas"]))
        samples = random.sample(pop, m)
        for sample in samples:
          if (len(theme_data[sample]["qas"]) > 0):
            q = random.choice(theme_data[sample]["qas"])["question"]
            unans_questions.append(q)

        if self.unans_filter == "tfidf":
          para = theme_data[i]["para"]
          vectorizer = TfidfVectorizer(analyzer="word", stop_words='english')
          cs, min_cs, rem = {}, np.inf, 0
          matrix = vectorizer.fit_transform([para] + 
                                            [qas["question"] for qas in theme_data[i]["qas"]] + 
                                            unans_questions)
          j = 1
          for cnt in range(len(theme_data[i]["qas"])):
            min_cs = min(min_cs, cosine_similarity(matrix[0], matrix[j])[0][0])
            j += 1
          for cnt in range(len(unans_questions)):
            x = cosine_similarity(matrix[0], matrix[j])[0][0]
            if x < 0.9*min_cs:
              que_to_add[i].append({"question": unans_questions[cnt], 
                                    "answer": "", "start_char": ""})
            else:
              rem += 1
          #print(rem, " out of", len(unans_questions) ," unans questions removed")

        elif self.unans_filter == "model":
          rem = 0
          outputs = self.qa_pipe(question=unans_questions,
                            context=[theme_data[i]["para"] for cnt in range(len(unans_questions))])
          if len(unans_questions)==1:
            outputs = [outputs]
          for idx, output in enumerate(outputs):
              para = theme_data[i]["para"]        
              if (output["answer"] == "") or (output["score"]<0.5):
                que_to_add[i].append({"question": unans_questions[idx],
                                          "answer": "", "start_char": ""})
              else:
                rem += 1
          #print(rem, " out of", len(unans_questions) ," unans questions removed")

      for i in range(n):
        for qa in que_to_add[i]:
          theme_data[i]["qas"].append(qa)

    def convert_to_csv(self,theme,theme_data):
      rows = []
      for data in theme_data:
        for qa in data["qas"]:
          sample = []
          sample.append(theme)
          sample.append(data["para"])
          sample.append(qa["question"])
          if qa["answer"] == '':
            sample.append(False)
            sample.append(str([]))
            sample.append(str([]))
          else:
            sample.append(True)
            sample.append(str([qa["answer"]]))
            sample.append(str([qa["start_char"]]))
          rows.append(sample)

      out = pd.DataFrame(rows)
      out.columns = ["Theme", "Paragraph", "Question", "Answer_possible", "Answer_text", "Answer_start"]
      return out
    
    
    @staticmethod
    def find_all(a_str, sub):
      start = 0
      while True:
          start = a_str.find(sub, start)
          if start == -1: return
          yield start
          start += len(sub) # use start += 1 to find overlapping matches

    def get_theme_dataset(self, para_df, qa_df, theme):
      all_para = para_df[para_df["theme"] == theme]["paragraph"].unique()

      theme_data = []
      for cnt, para in enumerate(tqdm(all_para)):
        qas = self.generate_qa(para)
        t = qa_df[qa_df["paragraph"] == para]
        for idx in t.index:
          if len(list(self.find_all(para, t.loc[idx]["answer"]))) == 1:
            qas.append({"question": t.loc[idx]["question"],
                        "answer": t.loc[idx]["answer"],
                        "start_char": para.find(t.loc[idx]["answer"])})
        theme_data.append({"para":para, "qas":qas})
        
      self.add_unans_que(theme_data)
      out = self.convert_to_csv(theme,theme_data)
      return out
    
    @staticmethod
    def parse_theme_name(x):
      return "".join(list(filter(lambda ch: "A"<=ch<="Z" or "a"<=ch<="z" or ch=="_" or "0"<=ch<="9", x)))
    
    def save_to_json(self,theme,df, output_dir):
      js = {}
      js["version"]= "2.1"
      js["data"] = []
      group_df = df.groupby("Paragraph")
      parajs = {}
      parajs["paragraphs"] = []
      context_useful = False
      for para in group_df.groups.keys():
          contjs = {}
          contjs["context"] = para
          contjs["qas"] = []
          questionable = False
          unique_que = group_df.get_group(para).groupby("Question")
          for que in unique_que.groups.keys():
              qasjs = {}
              qasjs["question"] = que
              ans = unique_que.get_group(que)
              qasjs["answers"] = []
              qasjs["id"] = str(ans.index[0])
              answerable=False
              for i in range(len(ans)):
                  if len(ans["Answer_start"].iloc[i])<=2:
                      qasjs["answers"].append({ 
                          "text": "",
                          "answer_start": 0,
                      })
                      continue
                  qasjs["answers"].append({ 
                      "text": ans["Answer_text"].iloc[i][2:-2],
                      "answer_start": int(ans["Answer_start"].iloc[i][1:-1]),
                  })
                  answerable=True

              contjs["qas"].append(qasjs)
              if answerable:
                  questionable = True
          parajs["paragraphs"].append(contjs)
          if questionable:
              context_useful = True
      if context_useful:
          js["data"].append(parajs)
          with open(output_dir+theme+"_train_data", "w") as f:
              json.dump(js, f)


import threading
class Worker(threading.Thread):
    def __init__(self,Id,  theme, qg_pipe, para_df, qa_df, theme_dataset, output_dir):
        threading.Thread.__init__(self)
        self.threadID = Id
        self.theme = theme
        self.qg_pipe = qg_pipe
        self.para_df = para_df
        self.qa_df = qa_df
        self.theme_dataset = theme_dataset
        self.output_dir = output_dir
        
    def run(self):
        self.theme_dataset[self.theme] = self.qg_pipe.get_theme_dataset(self.para_df,self.qa_df,self.theme).drop_duplicates()
        self.qg_pipe.save_to_json(self.theme,self.theme_dataset[self.theme], self.output_dir)
        #print(self.theme)
                
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="datasets/train_data.csv")
    parser.add_argument("--output_dir", type=str, default="oodomain_train/")
    parser.add_argument("--qa_model", type=str, default="vaibhav9/distil-roberta-qa")
    parser.add_argument("--use_cuda", action="store_false")
    parser.add_argument("--ner_limit", type=int, default=0)
    parser.add_argument("--save_csv", action="store_false")
    parser.add_argument("--use_qa_data", type=str, default="sample_question_answers.csv")
    args = parser.parse_args()
    
    para_df = pd.read_csv(args.input_dir)#id paragraph
    qa_df = pd.read_csv(args.use_qa_data)#question theme paragraph_id answer
    id2para = dict(zip(para_df["id"],para_df["paragraph"]))
    qa_df["paragraph"] = qa_df["paragraph_id"].map(id2para)
    
    qg_model = AutoModelForSeq2SeqLM.from_pretrained("valhalla/t5-small-qg-hl")
    qg_tokenizer = AutoTokenizer.from_pretrained("valhalla/t5-small-qg-hl")
    
    ## question theme paragraph_id answer
    
    
    qg_pipe = CustomPipeline(model = qg_model, 
                             tokenizer=qg_tokenizer, 
                             use_cuda=args.use_cuda,
                            qa_checkpoint = args.qa_model,
                             ner_limit = args.ner_limit,
                             
                            unans_filter="model")
    ##
    
    all_themes = para_df["theme"].unique()
    theme_dataset = {}
    total_start_time = time.time()
    
    threadLock = threading.Lock()
    import math
    n = 10
    for k in range(0, math.ceil(len(all_themes)/float(n))):
        i = n*k
        themes = []
        finetuned_model_paths = []
        threads = []
        # print("Batch_1")
        for j, theme in enumerate(all_themes[i:min(len(all_themes),i+n)]):
            thread = Worker(j, theme, qg_pipe, para_df, qa_df, theme_dataset, args.output_dir)
            thread.start()
            threads.append(thread)
        main_thread = threading.currentThread()
        threadLock.acquire()
        for t in threads:
            print(t)
            if t is not main_thread:
                t.join()
        threadLock.release()
      
    
    
    # for theme in all_themes:
    #     start_time = time.time()
    #     theme_dataset[theme] = qg_pipe.get_theme_dataset(para_df,qa_df,theme).drop_duplicates()
    #     qg_pipe.save_to_json(theme,theme_dataset[theme], args.output_dir)
    #     end_time = time.time()
    #     print("time_taken:", end_time - start_time)
    total_end_time = time.time()
    print("total time taken:", total_end_time - total_start_time)
    # out = pd.concat(list(theme_dataset.values()))
    # if args.save_csv:
    #   out.to_csv(args.output_dir + "synthetic_data.csv")
    

if __name__ == "__main__":
    main()
    
    
    
    