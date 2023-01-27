# Repository for project of Stanford cs224n class
## Project proposal
https://www.overleaf.com/read/vxyjkxsdpgsf

edit: https://www.overleaf.com/2969827368gmmqgvjygrsg

## How to use meta_train.py

### Step 1
In robustqa_meta/datasets, create two directories meta_train/ and meta_val/.

### Step 2
Copy all files in datasets/indomain_train and datasets/oodomain_train to meta_train/.

### Step 3
Copy all files in datasets/oodomain_val to meta_val/.

### Step 4
Move meta_train.py to the same directory as train.py.

### Step 5
Execute command `python meta_train.py --run-name meta_baseline --do-train --lr=3e-5 --meta-lr=1e-2 --meta-epochs=2400`.
