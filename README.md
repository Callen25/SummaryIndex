# Summary Indexing

This project attempts to use the Universal Sentence Encoder along with FAISS to index the summaries of documents as dense vectors in order to save memory + improve performance. This project consistst of two parts
1.  The standard vector index
2.  The hybrid index (combines results with SOLR index)

Both of these perform better than the SOLR baseline according to the TREC evaluation tool, but the hybrid index performs the best and outperforms the baseline in every category.

### Getting Started
To run the scripts in this project, create a python virtual environment (I used Python 3.7.2) and install the required packages with:
```sh
$ pip install -r requirements.txt
```

## Overview
Obtaining the results can be run in 3 steps: 
1. Parsing Data
2. Building the Index
3. Evaluation

For convenience, data is saved in the eval folder so that step 3 can be directly skipped to. (The first two steps are very CPU/GPU intensive and take a while)

### Parsing Data
In this step, the TREC XML files are parsed and sentences are transformed into sentence vectors with the Universal Sentence Encoder. For each document, only sentences with a certain similarity threshold to the headline vector are saved. A list of vectors for each document are saved in a pickle file so that consequent steps can use this data without needing to rerun this step.(*Note: this step requires trec data to be saved in a data/ directory) Run this step with:
```sh
$ python parse_data.py
```

### Building the Index
This step reads the vectors saved in the previous step and saves them to a serialized FAISS index as well as saving a mapping from doc-id's to faiss id's. Run this step with:
```sh
$ python build_index.py
```

### Evaluation
This step evaluates the vector model with different threshold and range values, as well as evaluates the hybrid model with different weights for the vector model.

1. First, generate results for the TREC evaluation tool. This is already done by default. 
* For Vector Model, run:
```sh
$ python eval.py
```
* For Hybrid Model, run:
```sh
$ python eval_supplement.py
```

2. Then, to evaluate the results with the TREC evaluation tool, navigate to the eval folder and run the bash script. (*Note: You must be in the eval directory for the bash script to work). This is already done by default.
* For Vector Model, run:
```sh
$ cd eval
$ ./getResults.sh
```
* For Hybrid Model, run:
```sh
$ cd eval
$ ./getSupResults.sh
```
3. To view a plot of the result, and get optimal values for each model, run:
 * For Vector Model, run:
```sh
$ python plot.py
```
* For Hybrid Model, run:
```sh
$ python sup_plot.py
```
## Results
The baseline results are save in eval/out.txt
The optimal vector model results are saved in eval/trec_output/range_0.95_thresh_0.4.txt
The optimal hybrid model results are saved in eval/trec_sup_output/weight_8.txt

