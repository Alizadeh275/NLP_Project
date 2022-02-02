# Training Adaptive Computation for Open-Domain Question Answering with Computational Constraints

This repository is the official implementation of [This Paper](https://arxiv.org/abs/2107.02102). 

>📋  Optional: include a graphic explaining your approach/main result, bibtex entry, link to demos, blog posts and tutorials

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

>📋  Describe how to set up the environment, e.g. pip/conda/docker commands, download datasets, etc...

1. Clone APE project from github:

```setup
!git clone https://github.com/uclnlp/APE
```

2. Clone FiD project from github:

```setup
!git clone https://github.com/facebookresearch/FiD.git
```

3. Download FiD data using follow bash command:

```setup
%%shell
bash FiD/get-data.sh
```

4. Split downloaded data to 8 parts (We will use one of part)

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
