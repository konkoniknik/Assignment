# Hiring Assessment: Text Normalization

In this assessment task we implement various normalization techniques for textual data. We are given
a csv file containing  unnormalized raw information about composition writers and the respective normalized information about the writers. We use rule based approaches mainly for preprocessing, and:
1. **Named Entity Recognition (NER)**: Identifies and extracts relevant entities from the data.
2. **Large Language Model (LLM)**: Utilizes a pre-trained language model for normalization.
3. **Custom Deep Learning Method**: A simple transformer-based approach trained on part of the dataset.


## ⚠️ Redundancies and Inconsistencies in the Dataset

We have observed several inconsistencies in the dataset that require attention:

#### Example 1:
According to the description, **Justyce Kaseem Wright** should be normalized as:  
- **RAW TEXT:** `<Unknown>/Wright, Justyce Kaseem`  
- **Normalized Text:** `Justyce Kaseem Wright`

However, in the dataset, we find:  
- **RAW TEXT:** `<Unknown>/Wright, Justyce Kaseem`  
- **Normalized Text:** `Wright/Justyce Kaseem`

#### Example 2:
In some cases, common Latin names are not parsed correctly. For instance:  
- **RAW TEXT:** `Mendel Brikman`  
- **Normalized Text:** *(empty)*  

These kinds of redundancies and inconsistencies are pervasive throughout the dataset.


### Conventions for Handling Inconsistencies

To address these issues, we propose the following conventions in our implementation:

1. **Label Adherence:**  
   When possible, we adhere to the provided labels and use a comma (`,`) as a delimiter to separate components.

2. **Latin Characters with Diacritics:**  
   Names containing Latin characters with diacritics are preserved as valid names.

3. **Non-Latin Characters:**  
   Names containing non-Latin characters are filtered out and excluded.

Using these conventions, the example **Mendel Brikman** would be included in the normalized text.


### Impact of Inconsistencies

Potential inconsistencies in the dataset can influence perceived performance metrics. For example:
- Including **Mendel Brikman** in the normalized text would reduce overall performance relative to the baseline, even though this behavior aligns with our conventions and expectations.

## Methods

In this assessment we try various different methods for the normalisation of the composition writers:

### Rule-based and NER(+POS) Implementation

In  this assessment, our main implementation is in Jupyter notebook `Normalisation_main.ipynb`. In this notebook, we read the input data from:

 - `'./data/raw/normalization_assesment_dataset_10k.csv'`

We preprocess the data by spliting into smaller phrases which are normalised writer candidates. For this split we use common separators ("weak" schema). We can also choose to use a more agressive separation schema ("strong") which utilizes the Counter class, a specialized dictionary subclass designed for counting hashable objects. However in  our current implementation we opt for the "weak" variant. Based on this separation we create lists of phrases which are candidates for composition writers. Note that no information has been used from the data until this step as such we perform it on the whole dataset.

We then split the data into train validation and test sets, and from the train set we find common words which are indicative of non-writer status and we can filter out. 

Next, now infering the normalized writers on new data, we filter out the common words identified in the previous steps and we also remove non-latin phrases (latin with diaritics are not removed). We output the results of this preprocessing on the test dataframe (for potential evaluation).

Finally, on the preprocessed data we perform NER using the `"en_core_web_trf"` model from spacy. We keep phrases including name entities (`ent.label_ == "PERSON"`) and filter out the rest. We then either drop the filtered out phrases (if precision is important), or we handle the phrases by performing Part of Speech tagging and droping phrases which dont include nouns or adjectives (if we want to strike a balance between precision and recall and drop only  phrases highly likely not to be names). 


#### Pros:
- **Customizable**: Easy to adjust the filtering rules for specific needs.
- **Resource Efficient**: Does not require extensive computational resources.
- **Interpretability**: The logic is transparent and easy to debug.

#### Cons:
- **Limited Coverage**: May miss valid writer names not recognized by rules or NER.
- **Language Dependence**: Heavily reliant on NER and POS tagging, which can fail for names with uncommon patterns.

### Custom Deep Learning Implementation

In Jupyter notebook `Normalisation_custom_dl.ipynb` we attempt a simpler technique, and separate into train validation and test sets as before, however we utilize Deep learning and implement a custom transformer (encoder) model with torch. We tokenize with character level tokenization  and train the model to directly perform  normalisation on the raw data, by utilizing the clean data we have as text.

#### Notes:

Unfortunately although this approach is promising, it was not able to properly converge (the output was in part appropriate, but included also irrelevant characters after the appropriate sequences on the output context). We likely need more data/compute, and maybe some modifications in the method (see next steps).


#### Pros:
- **Scalable**: Can adapt to larger datasets and complex patterns.
- **End-to-End**: Automates the normalization process without manual rules.

#### Cons:
- **Compute Intensive**: Requires significant data and computational resources for effective training.
- **Unstable**: Results depend on proper convergence, which may require hyperparameter tuning and additional iterations.


### LLM-based Implementation 

In Jupyter notebook `Normalisation_llm.ipynb` we utilize a LLM through groq. Specifically, we use `llama3-8b-8192`. We prompt the model and we send batches to it in order to perform normalisation on the batch. We concatenate the outputs after some post-processing to get the normalized text, and save it to a new dataframe.

#### Pros:
- **High Accuracy**: Leveraging a large language model (LLM) can handle complex name variations and ambiguous cases effectively.
- **Ease of Use**: Minimal setup required beyond LLM integration.

#### Cons:
- **Costly**: LLM inference can be expensive, especially for large datasets.
- **Black Box**: Less transparent compared to rule-based methods; difficult to interpret errors.


### Performances

To measure the performances we run the evaluation script `Evaluation.ipynb` on the respective file and column where we include the ouputs for the technique we want to evaluate. We measure, F1 score, precision, recall, Edit Distance  BLEU Score, and also (for completion) Accuracy, and the Confusion Matrix of each technique (NAME_OUT column convention) compared to the CLEAN_TEXT column of the respective test set (baseline):

#### Performance Comparison

The table below summarizes the performance metrics of each method:

| Method                   | Accuracy   | Precision  | Recall     | F1 Score   | Avg. Edit Distance | Avg. BLEU Score | Notes                                  |
|--------------------------|------------|------------|------------|------------|--------------------|-----------------|----------------------------------------|
| **Rule-based** (simple preprocessing) | 71.79%     | 80.42%     | 86.99%     | 83.58%     | 4.28              | 0.78           | Simple preprocessing and filtering.    |
| **NER Only** (r=0.25)    | 69.11%     | 84.37%     | 79.25%     | 81.73%     | 5.16              | 0.70           | Relies only on named entity recognition. |
| **NER & POS** (r=(0.25, 0.25)) | 71.99%     | 80.76%     | 86.90%     | 83.72%     | 4.25              | 0.78           | Combines NER with POS tagging for filtering. |
| **LLM** (Llama-3.1-8B) *(Small sample)* | 79.31%     | 88.46%     | 88.46%     | 88.46%     | 2.40              | 0.73           | Highly accurate; results based on a small sample. |
| **DL** *(Not Converged)* | 1.85%      | 2.13%      | 12.35%     | 3.63%      | 61.23             | 0.01           | Requires further training for meaningful results. |

#### Observations:
- **Rule-based** performs consistently well with high F1 scores and reasonable BLEU scores.
- **NER Only** achieves high precision but slightly lower recall, leading to lower F1 scores compared to the combined approach.
- **NER & POS** improves over NER-only by adding additional filtering, achieving the best balance of metrics among rule-based methods.
- **LLM** provides the highest accuracy and F1 score but is evaluated on a small sample and may be resource-intensive.
- **DL** underperforms significantly due to insufficient convergence, indicating a need for more training and possibly more data.


## Suggestions for Improvement

 There are several things that we can improve with the current implementations. First, regarding `Normalisation_main.ipynb`, we can extend the handler, and instead of using spacy POS as handler, we can use an LLM through the groq API. As generally API calls are more expensive than basic inference, we can use it only as a handler  after NER to mitigate cost. The code in this case would likely look similar to:

```python
....
client = Groq(api_key="",)
def filter_phrases_with_names_batch(phrases_list, r, handler="drop"):
    results = []
    # Process all phrases in the list as a batch and keep enabled only the ner component, 
    # for efficient processing based on https://spacy.io/usage/processing-pipelines
    docs = list(nlp.pipe(phrases_list))  #disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer"]))  # Batch process phrases with spaCy
    for i, doc in enumerate(docs):
        phrase = phrases_list[i]
        words = phrase.split()
        num_words = len(words)
        # Count "PERSON" entities in the processed doc
        num_names = sum(1 for ent in doc.ents if ent.label_ == "PERSON")
        
        # Check if the percentage of names meets the threshold
        if num_words > 0 and (num_names / num_words) >= r[0]:
            results.append(phrase)
        else:
            if handler == 'pos':
                num_nouns = sum(1 for token in doc if token.pos_ == "PROPN"  or token.pos_ == "NOUN" or token.pos_ == "ADJ" or token.pos_ == "INTJ")
                if num_words > 0 and (num_nouns / num_words) >= r[1]:
                    results.append(phrase)
            if handler == 'llm':
                chat_completion = client.chat.completions.create(messages=[{"role": "user","content": prompt, }],model="llama3-8b-8192",)
                output_msg = chat_completion.choices[0].message.content
                output_msg = postprocessing_function(output_msg)
                results.append(output_msg)
    return results
....
```

Additionally, we identified opportunities for improvement through hyperparameter tuning. For instance, performing a grid search on handlers combined with hyperparameter optimization for `r` could help refine the model. Evaluations on the validation set can guide the selection of the best configuration based on performance requirements.

For the custom Deep Learning implementation, several improvements can be made:
1. **Data Augmentation**: Collecting additional data could help the model generalize better.
2. **Scaling the Model**: Using a larger model with appropriate reparameterization might enhance performance.
3. **Handling Out-of-Vocabulary Cases**: Addressing out-of-vocabulary instances is critical, as they are currently ignored.
4. **Fixing Output Issues**: The current model outputs random symbols instead of spaces after correct parts of the output (e.g., `'john/////aaaaaa/aaaaaa'` instead of `'john           '`), which significantly impacts all performance metrics during evaluation. Adding an explicit regularization term in the loss function to encourage correct symbol placement may resolve this issue.

By addressing these aspects, we can improve the robustness and effectiveness of the deep learning approach.


## Functionality and Files
Here we describe inputs and outputs of each Notebook:
### Inputs & Outputs
- File: `Normalisation_main.ipynb`:
    - Cell 1: Splits phrases
        - Inputs: `'./data/raw/normalization_assesment_dataset_10k.csv'`
        - Ouput Cell 1: `./data/processed_data.csv` (contains all data, plus split candidate names as list of strings)
    - Cell 2: Creates the histogram of common words and splits the data int train|val|test
        - Inputs: `'./data/processed_data.csv'`
        - Intermediate Output(s) 2: `./data/processed_data_(train|val|test).csv`
    - Cell 3: Performs common irrelevant word removal and filters non-latin phrases
        - inputs: `./data/processed_data_test.csv`
        - outputs: `./data/processed_data_test.csv` (columns: `preproc_phrases` containing list of preprocessed phrases and `PREPROC_OUT` for evaluation of the preprocessed text)
    - Cell 4: NER and POS:
        - inputs: `./data/processed_data_test.csv` (column: `preproc_phrases`)
        - outputs: `./data/processed_data_test.csv` (columns: `f'ner_phrases_{handler}_{s}'` containing list of ner/pos phrases and `f"NER_OUT_{handler}_{s}"` for evaluation of the ner/pos text)

- File: `Normalisation_custom_dl.ipynb`
    - Inputs:  `'./data/raw/normalization_assesment_dataset_10k.csv'`
    - Outputs: `./data/output_file_dl.csv` (containing the same test set split as the main normalisation above since we used the same seed in the pandas split. Includes Raw, and Clean text, i.e., input and labels, plus the predictions as column `DL_OUT`)

- File: `Normalisation_llm.ipynb`
    - Inputs:  `'./data/raw/normalization_assesment_dataset_10k.csv'`
    - Outputs: `./data/output_file_llm.csv` (containing a small sample test set of 40 samples different from the original test split. Includes Raw, and Clean text, i.e., input and labels, plus the predictions as column `LLM_OUT`)

### Evaluation
- File: `Evaluation.ipynb`
    - Inputs: 
        - `'./data/raw/normalization_assesment_dataset_10k.csv'` column `CLEAN_TEXT` (labels)
        - Any file of the following: `./data/output_file_llm.csv` column `LLM_OUT`, `./data/output_file_dl.csv` column `DL_OUT`, `./data/processed_data_test.csv` column `PREPROC_OUT`or column `f"NER_OUT_{handler}_{s}"`
    - Outputs: 
        - Overall evaluation on the cell's output 
        - More detailed results on a per row basis on `./data/output_with_details.csv`



# Hiring Assessment: Cover Song Similarity

For this assessment we create a detailed description of our proposed implementation to design and evaluate a similarity metric that quantifies the relationship between original songs and their cover versions. We include the detailed desciption at:

 - `'./docs/Cover Song Similarity_ Detailed Description.pdf'`




