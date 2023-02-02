!python synthentic_gen.py --file test_pargraph.csv --output dataset/devrev_test


```.


├── README.md
├── args.py
├── augmentation.py
├── meta_train.py
├── oodomain_train
│   ├── duorc
│   ├── duorc_augmented
│   ├── duorc_augmented_synonym
│   ├── duorc_augmented_synonym_augmented
│   ├── race
│   ├── race_augmented
│   ├── race_augmented_synonym
│   ├── race_augmented_synonym_augmented
│   ├── relation_extraction
│   ├── relation_extraction_augmented
│   ├── relation_extraction_augmented_synonym
│   └── relation_extraction_augmented_synonym_augmented
├── pipeline.py
├── real_oodomain_train
│   ├── duorc_augmented
│   ├── race_augmented
│   └── relation_extraction_augmented
├── robustqa
│   ├── README.md
│   ├── args.py
│   ├── convert_to_squad.py
│   ├── datasets
│   │   ├── indomain_train
│   │   │   ├── nat_questions
│   │   │   ├── newsqa
│   │   │   └── squad
│   │   ├── indomain_val
│   │   │   ├── nat_questions
│   │   │   ├── newsqa
│   │   │   └── squad
│   │   ├── meta_train
│   │   │   ├── duorc
│   │   │   ├── nat_questions
│   │   │   ├── newsqa
│   │   │   ├── race
│   │   │   ├── relation_extraction
│   │   │   └── squad
│   │   ├── meta_val
│   │   │   ├── duorc
│   │   │   ├── race
│   │   │   └── relation_extraction
│   │   ├── oodomain_test
│   │   │   ├── duorc
│   │   │   ├── race
│   │   │   └── relation_extraction
│   │   ├── oodomain_train
│   │   │   ├── duorc
│   │   │   ├── race
│   │   │   └── relation_extraction
│   │   └── oodomain_val
│   │       ├── duorc
│   │       ├── race
│   │       └── relation_extraction
│   ├── environment.yml
│   ├── meta_train.py
│   ├── train.py
│   └── util.py
├── subsample.py
├── synonym_augmentation_test.ipynb
├── synthetic_gen.py
├── train.py
└── util.py

11 directories, 56 files
```
