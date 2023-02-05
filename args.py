import argparse

def get_train_test_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=64) # default 16
    parser.add_argument('--meta-epochs', type=int, default=1200)
    parser.add_argument('--num-epochs', type=int, default=3)
    parser.add_argument('--meta-lr', type=float, default=5e-5)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num-visuals', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save-dir', type=str, default='save/')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--train-datasets', type=str, default='squad,nat_questions,newsqa,duorc,race,relation_extraction')
    parser.add_argument('--run-name', type=str, default='multitask_distilbert')
    parser.add_argument('--recompute-features', action='store_true')
    parser.add_argument('--train-dir', type=str, default='datasets/devrev_train')
    parser.add_argument('--val-dir', type=str, default='datasets/devrev_val')
    parser.add_argument('--eval-dir', type=str, default='datasets/devrev_val')
    parser.add_argument('--eval-datasets', type=str, default='race,relation_extraction,duorc')
    parser.add_argument('--do-train', action='store_true')
    parser.add_argument('--do-eval', action='store_true')
    parser.add_argument('--sub-file', type=str, default='')
    parser.add_argument('--visualize-predictions', action='store_true')
    parser.add_argument('--eval-every', type=int, default=5000)
    args = parser.parse_args()
    return args

def get_reader_retriever_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reader_model", type=str, default="sentence-transformers/all-MiniLM-L12-v2")
    # change embedding size in QApipeline.py according to retriever model
    parser.add_argument("--retriever_model", type=str, default="vaibhav9/distil-roberta-qa")
    parser.add_argument('--embedding_size', type=int, default=386)
    parser.add_argument("--input_dir", type=str, default="datasets/train_data.csv")
    parser.add_argument("--output_dir", type=str, default="oodomain_train/")
    parser.add_argument("--use_cuda", action="store_false")
    parser.add_argument('--stride', type=int, default=128)
    parser.add_argument('--n_best_size', type=int, default=20)
    parser.add_argument('--n_clusters', type=int, default=1)
    parser.add_argument('--n_probe', type=int, default=1)
    parser.add_argument('--top_k', type=int, default=5)

    args = parser.parse_args()
    return args
