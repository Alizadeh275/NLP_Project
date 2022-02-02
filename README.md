# Training Adaptive Computation for Open-Domain Question Answering with Computational Constraints
 
In this repository we tried to compare two paper expriments: [FiD-base](https://arxiv.org/abs/2007.01282) and [APE-FiD-base](https://arxiv.org/abs/2107.02102). Those expriments was done on two datasets:


- NaturalQuestions
- TriviaQA

Result of  [APE-FiD-base](https://arxiv.org/abs/2107.02102) paper is shown in bottom:

![alt text](https://github.com/Alizadeh275/NLP_Project/blob/main/APE-base%20result/paper_results.PNG)

__Our goal is _reproduce_ [APE-FiD-base](https://arxiv.org/abs/2107.02102) paper results__.

## Requirements

To install all requirements go to notebooks folder and run this command:

```setup
pip install -r requirements.txt
```

## Setup

### 1. Clone [APE-FiD-base](https://arxiv.org/abs/2107.02102) project from github:

```setup
!git clone https://github.com/uclnlp/APE
```

### 2. Download APE data using follow bash command:
```setup
%%shell
bash APE/scripts/download-data.sh
```
#### Note: There is `no access` to this dataset and it was forbidden!
![alt text](https://github.com/Alizadeh275/NLP_Project/blob/main/APE-base%20result/forbidden_message.PNG)

### 3. Clone [FiD-base](https://arxiv.org/abs/2007.01282) project from github:

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


At the end of get-data.sh file, following codes made the RAM  full and preprocessing was stopped:

```setup
echo "Processing "$ROOT""
python src/preprocess.py $DOWNLOAD $ROOT
```

That is because when the FiD/src/preprocess.py file wants to load the open_domain/psgs_w100.tsv, RAM made full and execution was stopped:
(The psgs_w100.tsv is containing of passages and size of that is 10 GB)
```setup
if __name__ == "__main__":
    dir_path = Path(sys.argv[1])
    save_dir = Path(sys.argv[2])

    passages = util.load_passages(save_dir/'psgs_w100.tsv')
    passages = {p[0]: (p[1], p[2]) for p in passages}
```
### 5. So, we decide to split downloaded data to 8 equal parts (We will use first part for training):

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


### 6. Then we made preprocessing on selected part of data (change data path in FiD/src/preprocess.py):

```setup
if __name__ == "__main__":
    dir_path = Path(sys.argv[1])
    save_dir = Path(sys.argv[2])

    # passages = util.load_passages(save_dir/'psgs_w100.tsv')
    passages = util.load_passages('input1.csv')
    passages = {p[0]: (p[1], p[2]) for p in passages}
```

We do not consider all passages, so  we should consider index of existed passages,<br /> 
So we should change `select_examples_NQ` and `select_examples_TQA` methods in FiD/src/preprocess.py like this:<br /> (add this condition `if idx in passages`)

You should change FiD/src/util.py to load csv file (`reader = csv.reader(fin)`):
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
## Training

To train the model(s) in the paper, run this command:

```train
python train.py --input-data <path_to_data> --alpha 10 --beta 20
```

>📋  Describe how to train the models, with example commands on how to train the models in your paper, including the full training procedure and appropriate hyperparameters.

## Evaluation

To evaluate my model on ImageNet, run:

```eval
python eval.py --model-file mymodel.pth --benchmark imagenet
```

>📋  Describe how to evaluate the trained models on benchmarks reported in the paper, give commands that produce the results (section below).

## Pre-trained Models

You can download pretrained models here:

- [My awesome model](https://drive.google.com/mymodel.pth) trained on ImageNet using parameters x,y,z. 

>📋  Give a link to where/how the pretrained models can be downloaded and how they were trained (if applicable).  Alternatively you can have an additional column in your results table with a link to the models.

## Results

Our model achieves the following performance on :

### [Image Classification on ImageNet](https://paperswithcode.com/sota/image-classification-on-imagenet)

| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| My awesome model   |     85%         |      95%       |

>📋  Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 


## Contributing

>📋  Pick a licence and describe how to contribute to your code repository. 
