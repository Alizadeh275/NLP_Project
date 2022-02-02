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

1. Clone [APE-FiD-base](https://arxiv.org/abs/2107.02102) project from github:

```setup
!git clone https://github.com/uclnlp/APE
```

2. Download APE data using follow bash command:
```setup
%%shell
bash APE/scripts/download-data.sh
```
#### Note: There is `no access` to this dataset and it was forbidden!
![alt text](https://github.com/Alizadeh275/NLP_Project/blob/main/APE-base%20result/forbidden_message.PNG)

3. Clone [FiD-base](https://arxiv.org/abs/2007.01282) project from github:

```setup
!git clone https://github.com/facebookresearch/FiD.git
```

4. Download FiD data using follow bash command:

```setup
%%shell
bash FiD/get-data.sh
```

At the end of get-data.sh file, following codes made the RAM  full and preprocessing was stopped:

```setup
echo "Processing "$ROOT""
python src/preprocess.py $DOWNLOAD $ROOT
```

That is because when the FiD/src/preprocess.py file wants to load the open_domain/psgs_w100.tsv, RAM made full and execution was stopped:
(Size of psgs_w100.tsv is 10 GB)
```setup
if __name__ == "__main__":
    dir_path = Path(sys.argv[1])
    save_dir = Path(sys.argv[2])

    passages = util.load_passages(save_dir/'psgs_w100.tsv')
    passages = {p[0]: (p[1], p[2]) for p in passages}
```
5. Split downloaded data to 8 parts (We will use one of part)

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
