
import sys,getopt
import os
from os import listdir
from os.path import isfile, join
import warnings
# warnings.filterwarnings('ignore')
import torch
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import json
from transformers import BertForQuestionAnswering, BertTokenizerFast, AlbertForQuestionAnswering, AutoTokenizer, AutoModelForQuestionAnswering
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import yaml
import pickle5 as pickle
import threading

print(torch.cuda.device_count())

class SquadDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)

def check_if_model_exists(theme_model_path):
    files = os.listdir('theme_based_qna_models')
    if theme_model_path.split('/')[-1] in files:
        return True
    return False
    
def read_squad(path):
    with open(path, 'rb') as f:
        squad_dict = json.load(f)

    # initialize lists for contexts, questions, and answers
    contexts = []
    questions = []
    answers = []
    # iterate through all data in squad data
    for group in squad_dict['data']:
        for passage in group['paragraphs']:
            context = passage['context']
            for qa in passage['qas']:
                question = qa['question']
                if 'plausible_answers' in qa.keys():
                    access = 'plausible_answers'
                else:
                    access = 'answers'
                for answer in qa['answers']:
                    # append data to lists
                    contexts.append(context)
                    questions.append(question)
                    answers.append(answer)
    # return formatted data lists
    return contexts, questions, answers

def add_end_idx(answers, contexts):
    # loop through each answer-context pair
    for answer, context in zip(answers, contexts):
        # gold_text refers to the answer we are expecting to find in context
        gold_text = answer['text']
        # we already know the start index
        start_idx = answer['answer_start']
        # and ideally this would be the end index...
        end_idx = start_idx + len(gold_text)
        answer['answer_end'] = end_idx
        # ...however, sometimes squad answers are off by a character or two
        # if context[start_idx:end_idx] == gold_text:
        #     # if the answer is not off :)
        #     answer['answer_end'] = end_idx
        # else:
        #     for n in [1, 2]:
        #         if context[start_idx-n:end_idx-n] == gold_text:
        #             # this means the answer is off by 'n' tokens
        #             answer['answer_start'] = start_idx - n
        #             answer['answer_end'] = end_idx - n


def add_token_positions(encodings, answers, new_tokenizer):
    # initialize lists to contain the token indices of answer start/end
    start_positions = []
    end_positions = []
    for i in range(len(answers)):
        # append start/end token position using char_to_token method
        # print(answers[i])
        start_positions.append(encodings.char_to_token(i, answers[i]['answer_start']))
        end_positions.append(encodings.char_to_token(i, answers[i]['answer_end']))

        # if start position is None, the answer passage has been truncated
        if start_positions[-1] is None:
            start_positions[-1] = 1024 #new_tokenizer.model_max_length
        # end position cannot be found, char_to_token found space, so shift position until found
        shift = 1
        while end_positions[-1] is None:
            if answers[i]['answer_end']<shift:
                end_positions[-1] = 0
                break
            end_positions[-1] = encodings.char_to_token(i, answers[i]['answer_end'] - shift)
            shift += 1
    # update our encodings object with the new token-based start/end positions
    encodings.update({'start_positions': start_positions, 'end_positions': end_positions})

                    
def validate_one_epoch(loader,model,device=None):
    if device is None:
        device = torch.device('cuda')
    model.eval()
    # initialize testidation set data loader
    acc = []
    # loop through batches
    for batch in loader:
        # we don't need to calculate gradients as we're not training
        with torch.no_grad():
            # pull batched items from loader
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            # we will use true positions for accuracy calc
            start_true = batch['start_positions'].to(device)
            end_true = batch['end_positions'].to(device)
            # make predictions
            outputs = model(input_ids, attention_mask=attention_mask)
            # pull prediction tensors out and argmax to get predicted tokens
            start_pred = torch.argmax(outputs['start_logits'], dim=1)
            end_pred = torch.argmax(outputs['end_logits'], dim=1)
            # calculate accuracy for both and append to accuracy list
            acc.append(((start_pred == start_true).sum()/len(start_pred)).item())
            acc.append(((end_pred == end_true).sum()/len(end_pred)).item())
    # calculate average accuracy in total
    acc = sum(acc)/len(acc)
    return acc





def train_model_without_pretrain(dataset_path,model_path,tokenizer_path,hugging_face,epochs,batch_size,still_train=1, device=None):
    theme = dataset_path.split('/')[-1][:-11]
    existing_model_path = f'final_theme_based_qna_models/qna_model_{theme}.pt'
    if check_if_model_exists(existing_model_path) and not still_train:
        return 
    # if check_if_model_exists(existing_model_path):
    #     theme += '_new'
    
    # print("hugging_face", hugging_face, "tokenizer_path", tokenizer_path)
    if hugging_face:
        new_tokenizer = AutoTokenizer.from_pretrained(f'{tokenizer_path}')
    else:
        new_tokenizer = BertTokenizerFast(tokenizer_file=f'{tokenizer_path}')
    # print(new_tokenizer)
    # print(new_tokenizer.model_max_length)
    train_path = dataset_path
    train_contexts, train_questions, train_answers = read_squad(f'{train_path}')
    train_contexts,val_contexts,train_questions,val_questions,train_answers,val_answers = train_test_split(train_contexts, train_questions,
                                                               train_answers,test_size=0.1,random_state=69)
    assert len(train_questions)>len(val_questions)
    
    add_end_idx(train_answers, train_contexts)
    add_end_idx(val_answers, val_contexts)    
    train_encodings = new_tokenizer(train_contexts, train_questions, truncation=True, padding=True,max_length=512,return_tensors='pt')
    val_encodings = new_tokenizer(val_contexts, val_questions, truncation=True, padding=True,max_length=512,return_tensors='pt')
    add_token_positions(train_encodings, train_answers, new_tokenizer)
    add_token_positions(val_encodings, val_answers, new_tokenizer)
    train_dataset = SquadDataset(train_encodings)
    val_dataset = SquadDataset(val_encodings)
    
    # model_path = 'model_180'
    if hugging_face:
        model = AutoModelForQuestionAnswering.from_pretrained(f'{model_path}')
    else:
        model = AutoModelForQuestionAnswering.from_pretrained(f'{model_path}')

    val_loader = DataLoader(val_dataset, batch_size=16)
    
    if device is None:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    optim = AdamW(model.parameters(), lr=1e-3)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    model.to(device)
    
    # for params in model.bert.parameters():
    #   params.requires_grad = False
    
    for epoch in range(epochs):
        # set model to train mode
        model.train()
        # setup loop (we use tqdm for the progress bar)
        loop = tqdm(train_loader, leave=True)
        for batch in loop:
            # initialize calculated gradients (from prev step)
            optim.zero_grad()
            # pull all the tensor batches required for training
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            start_positions = batch['start_positions'].to(device)
            end_positions = batch['end_positions'].to(device)
            # train model on batch and return outputs (incl. loss)
            outputs = model(input_ids, attention_mask=attention_mask,
                            start_positions=start_positions,
                            end_positions=end_positions)
            # extract loss
            loss = outputs[0]
            # calculate loss for every parameter that needs grad update
            loss.backward()
            # update parameters
            optim.step()
            # print relevant info to progress bar
            loop.set_description(f'Epoch {epoch}')
            loop.set_postfix(loss=loss.item())

        model.eval()
        val_acc = validate_one_epoch(val_loader,model,device)
        #if val_acc>0.6:
        #    break
    
    
    model.eval()
    # initialize testidation set data loader
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    # initialize list to store accuracies
    acc = []
    # loop through batches
    for batch in val_loader:
        # we don't need to calculate gradients as we're not training
        with torch.no_grad():
            # pull batched items from loader
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            # we will use true positions for accuracy calc
            start_true = batch['start_positions'].to(device)
            end_true = batch['end_positions'].to(device)
            # make predictions
            outputs = model(input_ids, attention_mask=attention_mask)
            # pull prediction tensors out and argmax to get predicted tokens
            start_pred = torch.argmax(outputs['start_logits'], dim=1)
            end_pred = torch.argmax(outputs['end_logits'], dim=1)
            # calculate accuracy for both and append to accuracy list
            acc.append(((start_pred == start_true).sum()/len(start_pred)).item())
            acc.append(((end_pred == end_true).sum()/len(end_pred)).item())
    # calculate average accuracy in total
    acc = sum(acc)/len(acc)
    print(acc)
    
    finetuned_model_path = f"final_theme_based_qna_models/qna_model_{theme}.pt"
    
    torch.save(model, finetuned_model_path)
    
    return theme, finetuned_model_path

    
    
class Worker(threading.Thread):
    def __init__(self, threadID, name, dataset_path,model_path,tokenizer_path,hugging_face,epochs,batch_size,still_train, device, themes, finetuned_model_paths, threadLock):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.dataset_path = dataset_path
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.hugging_face = hugging_face
        self.epochs = epochs
        self.batch_size = batch_size
        self.still_train = still_train
        self.themes = themes
        self.finetuned_model_paths = finetuned_model_paths
        self.device = device
        self.threadLock = threadLock
     
    def run(self):
    # Get lock to synchronize threads
    # self.threadLock.acquire()
        theme, finetuned_model_path = train_model_without_pretrain(self.dataset_path,self.model_path,self.tokenizer_path,self.hugging_face,self.epochs,self.batch_size,1, self.device)
        self.themes.append(theme)
        self.finetuned_model_paths.append(finetuned_model_path)
      # Free lock to release next thread
    # self.threadLock.release()   
    
    
    
    
if __name__=="__main__":
#     opts,args = getopt.getopt(sys.argv[1:],'',["model_path=","num_tasks=","dataset_path=",'tokenizer_path='])
    
#     for opt,arg in opts:
#         # print(opt,arg)
#         if opt == '--model_path':
#             model_path = arg
#         elif opt == '--num_tasks':
#             num_tasks = arg
#         elif opt=='--dataset_path':
#             dataset_path = arg
#         elif opt=='--tokenizer_path':
#           tokenizer_path = arg
    
    with open('config.yaml','r') as fp:
        args = yaml.safe_load(fp)
      
    
    num_tasks = args['num_tasks']
    still_train = args['still_train']
    hugging_face = args['hugging_face']
    tokenizer_path = args['tokenizer_path']
    model_path = args['model_path']
    dataset_path = args['dataset_path']
    batch_size = args['batch_size']
    epochs = args['epochs']
    dataset_dir = args['dataset_dir']

    seed = 7
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")
    
    themes_path = []
    for f in listdir(dataset_dir):
        if isfile(join(dataset_dir, f)) and not f.endswith(".pt"):
            themes_path.append(f)
        
    # themes_path= [f for f in listdir(dataset_dir) if isfile(join(dataset_dir, f))]
    # print(themes_path)
    
    theme_models_dict = {}

    # for task in range(num_tasks):
    #     path = '/home/ug2019/eee/19085023/' + dataset_path[task]
    #     theme, finetuned_model_path = train_model_without_pretrain(path,model_path,tokenizer_path,hugging_face,epochs,batch_size,1)
    #     theme_models_dict[theme] = finetuned_model_path
    
    threadLock = threading.Lock()
    import math
    for k in range(0, math.ceil(len(themes_path)/4.0)):
        i = 4*k
        themes = []
        finetuned_model_paths = []
        threads = []
        # print("Batch_1")
        for j, theme_path in enumerate(themes_path[i:min(len(themes_path),i+4)]):
            path = dataset_dir + '/' + theme_path
            print(theme_path)
            thread = Worker(j, theme_path, path,model_path,tokenizer_path,hugging_face,epochs,batch_size,1, 'cuda:'+str(j), themes, finetuned_model_paths, threadLock)
            thread.start()
            threads.append(thread)
        main_thread = threading.currentThread()
        threadLock.acquire()
        for t in threads:
            print(t)
            if t is not main_thread:
                t.join()
        threadLock.release()
      
        for theme,finetuned_model_path in zip(themes, finetuned_model_paths):
            theme_models_dict[theme] = finetuned_model_path
    
    # for theme_path in themes_path:
    #     print(theme_path)
    #     path = dataset_dir + '/' + theme_path
    #     theme, finetuned_model_path = train_model_without_pretrain(path,model_path,tokenizer_path,hugging_face,epochs,batch_size,1)
    #     theme_models_dict[theme] = finetuned_model_path
    
    
#     with open('/home/ug2019/eee/19085023/theme_based_bert/train_theme_based_qna_models/train_theme_models_dict.pickle', 'rb') as handle:
#       data=pickle.load(handle)
    
#     data.update(theme_models_dict)
        
    with open('train_theme_based_qna_models/final_theme_models_dict.pickle', 'wb') as handle:
        pickle.dump(theme_models_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)