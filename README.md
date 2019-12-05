# Topic-Modeling-and-Sentiment-Analysis
To run the api:

1. Clone the repo
2. In terminal, run 
```bash
conda env create -f environment.yml
```
3. Download Bert model from https://www.dropbox.com/sh/kdmz38dk77s3jr4/AABmgExe_56_OUnewlkKyz0pa?dl=0  and put all files under a folder named bert
4. In terminal run
```bash
conda activate bert
```
then run
```bash
python -m spacy download en_core_web_sm
```
5. In the root folder of terminal, run 
```bash
python app.py
```
6. Open a tab in a browser, run 
```bash
http://0.0.0.0:5000/predict?text=Trump is awesome;Hillary is Bad&model=rf
```
PS: model has options: 1.) rf (random forest) 2.) lstm 3.) bert
