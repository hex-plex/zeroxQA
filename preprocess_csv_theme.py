import pandas as pd
import json
import argparse


def main(args):
    df = pd.read_csv(args.data_csv)
    df["Theme"] = df["Theme"].apply(lambda x: "".join(list(filter(lambda s: "a"<=s<="z" or "A"<=s<="Z" or s=="_", x))))
    theme_df = df.groupby("Theme")
    for group in theme_df.groups.keys():
        js = {}
        js["version"]= "2.1"
        js["data"] = []
        group_df = theme_df.get_group(group).groupby("Paragraph")
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
            with open("devrev_train/"+group+"_train_data", "w") as f:
                json.dump(js, f)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-csv', type=str, default='train_data')
    args = parser.parse_args()
    main(args)
