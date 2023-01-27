import json
from pathlib import Path

import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
import nlpaug.flow as nafc

from nlpaug.util import Action

def find_stop_index(sorted_changes, value):
    for i in range(len(sorted_changes)):
        if value < sorted_changes[i]['orig_start_pos']:
            return i
    return len(sorted_changes)

def find_replaced_word(old_start_index, text, sorted_changes):
    old_end_index = old_start_index + len(text)
    delta = 0
    for change in sorted_changes:
        if old_start_index <= change['orig_start_pos'] < old_end_index:
            text = text[:change['orig_start_pos'] + delta] + change['new_token'] + \
                   text[change['orig_start_pos'] + len(change['orig_token']) + delta:]
            delta += len(change['new_token']) - len(change['orig_token'])
    return text


def process(contexts, questions, answer_starts, texts, answer_num_in_contexts, js):
    new_contexts = []
    new_questions = questions
    new_answer_starts = []
    new_texts = []
    aug = naw.SynonymAug(aug_src='wordnet', lang='eng', aug_min=1, aug_max=30, aug_p=0.3)
    # aug = naw.ContextualWordEmbsAug(model_path='bert-base-uncased')
    cursor = 0
    for i in range(len(contexts)):
        augmented_context, change_log = aug.augment(contexts[i])
        new_contexts.append(augmented_context)
        change_log.sort(key=lambda x: x['orig_start_pos'], reverse=False)
        for j in range(answer_num_in_contexts[i]):
            idx = find_stop_index(change_log, answer_starts[cursor])
            total_delta = 0
            for k in range(idx):
                total_delta += change_log[k]['new_start_pos'] - change_log[k]['orig_start_pos']
            new_answer_starts.append(answer_starts[cursor] + total_delta)
            new_texts.append(find_replaced_word(answer_starts[cursor], texts[cursor], change_log))
            cursor += 1
    return new_contexts, new_questions, new_answer_starts, new_texts

# ['data'][0]['paragraphs'][0]
# - ['context']
# - ['qas'][0]
#   - ['questions']
#   - ['answers'][0]
#     - ['text']
#     - ['answer_start']

def read_and_write(js):
    # 1D, context where the model finds answer from.
    contexts = []
    # 1D, for each context, there is a list of questions regarding this context.
    questions = []
    # 1D, for each question, there is a list of potential answers. Each of the answer has a character level starting
    # index in the context.
    answer_starts = []
    # 1D, for each question, there is a list of potential answers. This is the text of the answer.
    texts = []
    # 1D
    answer_num_in_contexts = []

    for x in js['data']:
        for y in x['paragraphs']:
            contexts.append(y['context'])
            cursor = 0
            for z in y['qas']:
                questions.append(z['question'])
                for a in z['answers']:
                    answer_starts.append(a['answer_start'])
                    texts.append(a['text'])
                    cursor += 1
            answer_num_in_contexts.append(cursor)

    new_contexts, new_questions, new_answer_starts, new_texts = \
        process(contexts, questions, answer_starts, texts, answer_num_in_contexts, js)

    i, j = 0, 0
    for x in range(len(js['data'])):
        for y in range(len(js['data'][x]['paragraphs'])):
            js['data'][x]['paragraphs'][y]['context'] = new_contexts[i]
            for z in range(len(js['data'][x]['paragraphs'][y]['qas'])):
                for a in range(len(js['data'][x]['paragraphs'][y]['qas'][z]['answers'])):
                    js['data'][x]['paragraphs'][y]['qas'][z]['answers'][a]['answer_start'] = new_answer_starts[j]
                    js['data'][x]['paragraphs'][y]['qas'][z]['answers'][a]['text'] = new_texts[j]
                    j += 1
            i += 1

    return js

def wrapper(path, aug_num):
    f = open(path)
    js = json.load(f)
    f.close()
    for i in range(aug_num):
        print("======================== Augmentation", i, "========================")
        f = open(path)
        new_js = json.load(f)
        f.close()
        new_js = read_and_write(new_js)
        for data in new_js['data']:
            js['data'].append(data)

    outfile = open(str(path) + "_augmented", 'w')
    json.dump(js, outfile)

def main():
    print("Generating augment data")
    file_paths = Path('oodomain_train').glob('*')
    for file_path in file_paths:
        if str(file_path).find('augmented') != -1: continue
        print("Processing: " + str(file_path))
        wrapper(file_path, aug_num=19)

if __name__ == '__main__':
    main()