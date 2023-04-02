# ZeroxQA: Few-shot Learning for QA Tasks

A simple QA task here is defined as a Comprehensive reading task. Where a paragraph is given containing information about a domain. Along with it, a question is given which can be answered using the paragraph and the supervision of starting and ending indexes in the paragraph corresponding to that question. A state-of-the-art approach for such a problem is employing a transformer while masking the question and using two classification heads on top of the transformers for predicting start and stop indices.

![Reader](media/reader.png)

The major constraints and challenges we faced are below:
- To have a model with very low inference time.
- To generalize the model over extremely unrelated domains.

Hence, we employ a Meta Learning-based training framework for training a shallow transformer model to generalize in out-of-domain queries while still offering faster convergence on out-of-domain targets.

![](media/meta_learning.png)

Hence, the above is the proposed framework along with [Bert-Mini-5](https://huggingface.co/mrm8488/bert-mini-5-finetuned-squadv2). For individual domains, the meta-model is later fine-tuned for inference.

## Usage

### Preparing Dataset

```bash
$ python preprocess_csv_theme.py --data-csv=train_data.csv
$ python synthetic_gen.py --input_dir=datasets/train_data.csv --ner_limit=2 --use_qa_data=sample_question_answers.csv
```

This preprocesses the whole dataset into different shards of datasets for different domains. Then synthetically generates question and answer pairs for unannotated paragraphs for improving in-domain queries.

## Training 

```bash
$ python meta_train.py --run-name meta_baseline --do-train --lr=1e-4 --meta-lr=5e-3 --meta-epochs=10000
$ python finetune.py
```
The first command trains a meta-model using MAML for the in-domain targets and the code is parallelized for training individual models on individual in-domain sets. The second command fine-tunes the meta-model on an out-of-domain query using synthetically generated datasets using the paragraphs for the out-of-domain queries.

## Quantization

We performed model quantization and graph optimization using ONNX to compare the runtime optimization on two models, namely small-Electra and mini5-BERT, as shown in the figure. Using ONNX runtime optimizations, mini5-BERT only took 120 milliseconds and small-Electra took only 140 milliseconds. We observed that mini5-BERT offered us the best tradeoff between speed and accuracy.

Model Quantization:
- Converts 32-bit floating point tensors and operations in the model to 8-bit integer values.
- A quantized model executes some or all operations on tensors with reduced precision rather than full precision 32-bit floating point values.
- Allows for a compact model representation and the use of high-performance vectorized operations.
- We used the ORTQuantizer module of ONNX (Open Neural Network eXchange) module of the Optimum Library.

Graph Optimization:
- Semantics-preserving graph rewrites which remove redundant nodes and redundant computation by pruning the computational graph of the model.
- These optimizations also include fusion of nodes and constant folding.

## Results

One can observe the improvement due to the meta-learned model on out-of-domain queries.
![](media/plot.png)

The Improvement for quantizing the model can be observed with the the distribution of average inference time for different settings.

![](media/mini5.png)
![](media/mini55.png)
