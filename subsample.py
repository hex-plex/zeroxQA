import json
import random
import copy
from pathlib import Path

def subsample(data_file, subsample_num):
    squad_data = json.load(open(data_file, 'rb'))
    # Count number of questions
    all_ids = []
    for example in squad_data["data"]:
        for paragraph in example["paragraphs"]:
            for question_answer in paragraph["qas"]:
                all_ids.append(question_answer['id'])

    chosen_ids = set(random.sample(all_ids, k=subsample_num))
    subsampled_data = copy.deepcopy(squad_data)
    # Verify the data
    for article in subsampled_data["data"]:
        for paragraph in article["paragraphs"]:
            new_qas = []
            for qas in paragraph["qas"]:
                if qas['id'] in chosen_ids:
                    new_qas.append(qas)
            paragraph['qas'] = new_qas

    with open(str(data_file) + '_subsampled', "w") as output_file:
        json.dump(subsampled_data, output_file)

def count(data_file):
    squad_data = json.load(open(data_file, 'rb'))
    # Count number of questions
    all_ids = []
    for example in squad_data["data"]:
        for paragraph in example["paragraphs"]:
            for question_answer in paragraph["qas"]:
                all_ids.append(question_answer['id'])
    print("Number of entries in %s: %s" % (str(data_file), len(all_ids)))

if __name__ == '__main__':
    file_paths = Path('datasets/indomain_train').glob('*')
    for file_path in file_paths:
        count(file_path)
        subsample(file_path,1000)