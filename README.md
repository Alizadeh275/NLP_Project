# Training Adaptive Computation for Open-Domain Question Answering with Computational Constraints
 
In this repository we tried to compare two paper expriments: [FiD-base](https://arxiv.org/abs/2007.01282) and [APE-FiD-base](https://arxiv.org/abs/2107.02102). Those expriments was done on two datasets:


- NaturalQuestions
- TriviaQA

Result of  [APE-FiD-base](https://arxiv.org/abs/2107.02102) paper is shown in bottom:

![alt text](https://github.com/Alizadeh275/NLP_Project/blob/main/screenshots/paper_results.PNG)

__Our goal is _reproduce_ [APE-FiD-base](https://arxiv.org/abs/2107.02102) paper results__.

## Requirements

To install all requirements go to notebooks folder and run this command:

```setup
pip install -r requirements.txt
```

## Setup

### 1. Clone [APE-FiD-base](https://github.com/uclnlp/APE) project from github:

```setup
!git clone https://github.com/uclnlp/APE
```

### 2. Download APE data using follow bash command:
```setup
%%shell
bash APE/scripts/download-data.sh
```
#### Note: There is `no access` to this dataset and it was forbidden!
![alt text](https://github.com/Alizadeh275/NLP_Project/blob/main/screenshots/forbidden_message.PNG)

### 3. Clone [FiD-base](https://github.com/facebookresearch/FiD.git) project from github:

```setup
!git clone https://github.com/facebookresearch/FiD.git
```

### 4. Download FiD data using follow bash command:

```setup
%%shell
bash FiD/get-data.sh
```


### Data format

The expected data format is a list of entry examples, where each entry example is a dictionary containing
- `id`: example id, optional
- `question`: question text
- `target`: answer used for model training, if not given, the target is randomly sampled from the 'answers' list
- `answers`: list of answer text for evaluation, also used for training if target is not given
- `ctxs`: a list of passages where each item is a dictionary containing
        - `title`: article title
        - `text`: passage text

Entry example:
```
{
  'id': '0',
  'question': 'What element did Marie Curie name after her native land?',
  'target': 'Polonium',
  'answers': ['Polonium', 'Po (chemical element)', 'Po'],
  'ctxs': [
            {
                "title": "Marie Curie",
                "text": "them on visits to Poland. She named the first chemical element that she discovered in 1898 \"polonium\", after her native country. Marie Curie died in 1934, aged 66, at a sanatorium in Sancellemoz (Haute-Savoie), France, of aplastic anemia from exposure to radiation in the course of her scientific research and in the course of her radiological work at field hospitals during World War I. Maria Sk\u0142odowska was born in Warsaw, in Congress Poland in the Russian Empire, on 7 November 1867, the fifth and youngest child of well-known teachers Bronis\u0142awa, \"n\u00e9e\" Boguska, and W\u0142adys\u0142aw Sk\u0142odowski. The elder siblings of Maria"
            },
            {
                "title": "Marie Curie",
                "text": "was present in such minute quantities that they would eventually have to process tons of the ore. In July 1898, Curie and her husband published a joint paper announcing the existence of an element which they named \"polonium\", in honour of her native Poland, which would for another twenty years remain partitioned among three empires (Russian, Austrian, and Prussian). On 26 December 1898, the Curies announced the existence of a second element, which they named \"radium\", from the Latin word for \"ray\". In the course of their research, they also coined the word \"radioactivity\". To prove their discoveries beyond any"
            }
          ]
}
```

> Note: Data files are seprate from each other and [preprocess.py](https://github.com/Alizadeh275/NLP_Project/blob/main/templates/FiD-base/src/preprocess.py) gathered all sepreted data and form all data according to above data format
At the end of get-data.sh file, following codes made the RAM  full and preprocessing was stopped:

```setup
echo "Processing "$ROOT""
python src/preprocess.py $DOWNLOAD $ROOT
```

That is because when the [preprocess.py](https://github.com/Alizadeh275/NLP_Project/blob/main/templates/FiD-base/src/preprocess.py) file wants to load the open_domain/psgs_w100.tsv, RAM made full and execution was stopped:
(The psgs_w100.tsv is containing of passages and size of that is 10 GB)
```setup
if __name__ == "__main__":
    dir_path = Path(sys.argv[1])
    save_dir = Path(sys.argv[2])

    passages = util.load_passages(save_dir/'psgs_w100.tsv')
    passages = {p[0]: (p[1], p[2]) for p in passages}
```
### 5. So, we decide to split passages data to 8 equal parts (We will use first part for training):

```setup
import pandas as pd

in_csv = '/content/open_domain_data/psgs_w100.tsv'

number_lines = sum(1 for row in (open(in_csv)))

rowsize = round(number_lines / 8)

for i in range(1,number_lines,rowsize):

    df = pd.read_csv(in_csv, header=None, nrows = rowsize, skiprows = i , delimiter='\t')
    out_csv = 'input' + str(i) + '.csv'

    df.to_csv(out_csv,
          index=False,
          header=False,
          mode='a',#append data to csv file
          chunksize=rowsize)#size of data to append for each loop
```


### 6. Then we made preprocessing on selected part of data (change data path in [preprocess.py](https://github.com/Alizadeh275/NLP_Project/blob/main/templates/FiD-base/src/preprocess.py)):

```setup
if __name__ == "__main__":
    dir_path = Path(sys.argv[1])
    save_dir = Path(sys.argv[2])

    # passages = util.load_passages(save_dir/'psgs_w100.tsv')
    passages = util.load_passages('input1.csv')
    passages = {p[0]: (p[1], p[2]) for p in passages}
```

We do not consider all passages, so  we should consider index of existed passages,<br /> 
So we should change `select_examples_NQ` and `select_examples_TQA` methods in [preprocess.py](https://github.com/Alizadeh275/NLP_Project/blob/main/templates/FiD-base/src/preprocess.py) like this:<br /> (add this condition `if idx in passages`)

`select_examples_TQA:`
```setup
def select_examples_TQA(data, index, passages, passages_index):
    selected_data = []
    for i, k in enumerate(index):
        ex = data[k]
        q = ex['Question']
        answers = ex['Answer']['Aliases']
        target = ex['Answer']['Value']

        ctxs = [
                  {
                    
                      'id': idx,
                      'title': passages[idx][1],
                      'text': passages[idx][0],
                  }
                  for idx in passages_index[ex['QuestionId']] if idx in passages
                  
              ]

        if target.isupper():
            target = target.title()
        selected_data.append(
            {
                'question': q,
                'answers': answers,
                'target': target,
                'ctxs': ctxs,
            }
        )
    return selected_data
```

`select_examples_NQ:`
```setup
def select_examples_NQ(data, index, passages, passages_index):
    selected_data = []
    for i, k in enumerate(index):
        ctxs = [
                {
                    'id': idx,
                    'title': passages[idx][1],
                    'text': passages[idx][0],
                }
                for idx in passages_index[str(i)] if idx in passages
            ]
        dico = {
            'question': data[k]['question'],
            'answers': data[k]['answer'],
            'ctxs': ctxs,
        }
        selected_data.append(dico)

    return selected_data
```

You should change [util.py](https://github.com/Alizadeh275/NLP_Project/blob/main/templates/FiD-base/src/util.py) to load csv file (`reader = csv.reader(fin)`):
```setup
def load_passages(path):
    if not os.path.exists(path):
        logger.info(f'{path} does not exist')
        return
    logger.info(f'Loading passages from: {path}')
    passages = []
    with open(path) as fin:
        reader = csv.reader(fin)
        for k, row in enumerate(reader):
            if not row[0] == 'id':
                try:
                    passages.append((row[0], row[1], row[2]))
                except:
                    logger.warning(f'The following input line has not been correctly loaded: {row}')
    return passages
```

But we see RAM error again when loading data, So we decide to run NQ and TQA seperately, So we comment Trivia Loading segment in [preprocess.py](https://github.com/Alizadeh275/NLP_Project/blob/main/templates/FiD-base/src/preprocess.py) :
```setup
  # #load Trivia question idx
    # TQA_idx, TQA_passages = {}, {}
    # for split in ['train', 'dev', 'test']:
    #     with open(dir_path/('TQA.' + split + '.idx.json'), 'r') as fin:
    #         TQA_idx[split] = json.load(fin)
    #     with open(dir_path/'tqa_passages' /  (split + '.json'), 'r') as fin:
    #         TQA_passages[split] = json.load(fin)


    # originaltrain, originaldev = [], []
    # with open(dir_path/'triviaqa-unfiltered'/'unfiltered-web-train.json') as fin:
    #     originaltrain = json.load(fin)['Data']
    
    # with open(dir_path/'triviaqa-unfiltered'/'unfiltered-web-dev.json') as fin:
    #     originaldev = json.load(fin)['Data']

    # TQA_train = select_examples_TQA(originaltrain, TQA_idx['train'], passages, TQA_passages['train'])
    # TQA_dev = select_examples_TQA(originaltrain, TQA_idx['dev'], passages, TQA_passages['dev'])
    # TQA_test = select_examples_TQA(originaldev, TQA_idx['test'], passages, TQA_passages['test'])
   
    # TQA_save_path = save_dir / 'TQA'
    # TQA_save_path.mkdir(parents=True, exist_ok=True)

    # with open(TQA_save_path/'train.json', 'w') as fout:
    #     json.dump(TQA_train, fout, indent=4)
    # with open(TQA_save_path/'dev.json', 'w') as fout:
    #     json.dump(TQA_dev, fout, indent=4)
    # with open(TQA_save_path/'test.json', 'w') as fout:
    #     json.dump(TQA_test, fout, indent=4)

```

and continue preprocessing on NQ data.
<br />
After running of preprocess.py, three files created in open_domain_data/NQ folder:
- train.py
- test.py
- dev.py

## Train (APE-FiD-base)

[`train.py`](https://github.com/Alizadeh275/NLP_Project/blob/main/templates/APE-FiD-base/FiD/train.py) provides the code for training a model from scratch. An example usage of the script with some options is given below:

```shell
%%shell

python APE/FiD/train.py \
  --checkpoint_dir checkpoint \
  --train_data_path open_domain_data/NQ/train.json \
  --dev_data_path open_domain_data/NQ/dev.json \
  --model_size base \
  --per_gpu_batch_size 4 \
  --n_context 1 \
  --name my_experiment \
  --eval_freq 1000
```  


## Evaluation (APE-FiD-base)

[`test.py`](https://github.com/Alizadeh275/NLP_Project/blob/main/templates/APE-FiD-base/FiD/test.py) provides the script to evaluate the performance of the model. An example usage of the script is provided below.

```eval
%%shell

python APE/FiD/test.py \
  --model_path pretrained_models/nq_reader_base \
  --test_data_path open_domain_data/NQ/test.json \
  --model_size base \
  --per_gpu_batch_size 4 \
  --n_context 1 \
  --name my_test \
  --checkpoint_dir checkpoint
```

> Test Result:
```eval
[02/01/2022 19:59:57] {test.py:127} INFO - Start eval
[02/01/2022 20:22:01] {test.py:65} INFO - total number of example 3610
[02/01/2022 20:22:01] {test.py:136} INFO - EM 39.128542
```

> Conclusion: <br /> The reached EM is lower than APE-FiD-base result, and we think it is because that we do expriments on small parts of data (due to RAM limitation).



Now we examine FiD-base project. First train the model and then evaluate it.

## Train (FiD-base)

[`train_reader.py`](https://github.com/Alizadeh275/NLP_Project/blob/main/templates/FiD-base/train_reader.py) provides the code for training a model from scratch. An example usage of the script with some options is given below:

```shell
%%shell
python FiD/train_reader.py \
        --train_data open_domain_data/NQ/train.json \
        --eval_data open_domain_data/NQ/dev.json \
        --model_size base \
        --per_gpu_batch_size 1 \
        --n_context 1 \
        --name my_experiment \
        --checkpoint_dir checkpoint \
        --use_checkpoint 
```     

## Evaluation (FiD-base)

[`test_reader.py`](https://github.com/Alizadeh275/NLP_Project/blob/main/templates/FiD-base/test_reader.py) provides the script to evaluate the performance of the model. An example usage of the script is provided below.

```eval
%%shell



python FiD/test_reader.py \
  --model_path pretrained_models/nq_reader_base \
  --eval_data open_domain_data/NQ/test.json \
  --model_size base \
  --per_gpu_batch_size 1 \
  --n_context 100 \
  --name my_test \
  --checkpoint_dir checkpoint
```


> Test Result:
```result
[02/02/2022 19:06:33] {test_reader.py:128} INFO - Start eval
[02/02/2022 19:23:47] {test_reader.py:136} INFO - EM 35.86, Total number of example 3610
```
> Conclusion: <br /> The reached EM is lower than FiD-base result, and we think it is because that we do expriments on small parts of data (due to RAM limitation).



## Results

> Exact match scores on NaturalQuestions  test sets.

| Model name         | EM on NaturalQuestions (test set)  | Evaluation Time |
| ------------------ |---------------- |---------------- |
| APE-FiD-base   |     39.128542 %         | 23 Min |
| FiD-base   |     35.86 %         | 17 Min |

> 📋 __Conclusion__: We can see that APE-FiD-base EM is better than the FiD-base (like [APE-FiD-base](https://arxiv.org/abs/2107.02102) paper results), but FiD-base runtime is lower that APE-FiD-base runtime.<br /><br />
> 📋 __Note__:The results we obtained are lower than the results of the original paper (due to memory constraints we have to experiment on a small portion of the data) and therefore we expected the results to be different from the results of the original paper.
Of course, our results showed the superiority of the proposed method of the original paper, and we observed in our experiments this superiority of the proposed method in the test set data.


## Contributing

>📋  You can use another dataset (TriviaQA) to compare FiD-base and APE-FiD-base
