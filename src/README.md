## Celadon Training 

This file describes the training process for the Celadon classifier, as completed on the Jean-Zay computing cluster. 
If you are looking for applications of Celadon, please navigate to the `/scripts` directory. If you want to replicate
Celadon on your own data, feel free to use these files as reference for your own specialized application. 

### 1. Dataset Curation
Given the incredible linguistic diversity of Common Corpus, it was important that Celadon was compatible with many different
languages. The linguistic split within the dataset is approximately proportional to the linguistic split within the PleIAs family
of language models. The exact dataset split is as follows, for a total of 2 million samples:

| Language     | Dataset                | # Samples  |
|--------------|------------------------|------------|
| English   | [US-PD-Newspapers](https://huggingface.co/datasets/PleIAs/US-PD-Newspapers)              |    250K        |
| English   | [US-PD-Books](https://huggingface.co/datasets/storytracer/US-PD-Books)             |      250K      |
| French   | [French-PD-Newspapers](https://huggingface.co/datasets/PleIAs/French-PD-Newspapers)             |    250K        |
| French   | [French-PD-Books](https://huggingface.co/datasets/PleIAs/French-PD-Books)               |    250K        |
| German   | [German-PD](https://huggingface.co/datasets/PleIAs/German-PD)               |     100K       |
| German   | [German-PD-Newspapers](https://huggingface.co/datasets/storytracer/German-PD-Newspapers)               |   100K         |
| Spanish   | [Spanish-PD-Newspapers](https://huggingface.co/datasets/PleIAs/Spanish-PD-Newspapers)               |   100K         |
| Spanish   | [Spanish-PD-Books](https://huggingface.co/datasets/PleIAs/Spanish-PD-Books)               |     100K       |
| Italian  | [Italian-PD](https://huggingface.co/datasets/PleIAs/Italian-PD)               |     200K       |
| Dutch  | [Dutch-PD](https://huggingface.co/datasets/PleIAs/Dutch-PD)              |     100K       |
| Portuguese  | [Portuguese-PD](https://huggingface.co/datasets/PleIAs/Portuguese-PD)              |     100K       |
| Latin  | [Latin-PD](https://huggingface.co/datasets/PleIAs/Latin-PD)              |     100K       |
| Polish  | [Polish-PD](https://huggingface.co/datasets/PleIAs/Polish-PD)              |      100K      |

### 2. Annotation Creation
In essence, Celadon is a classifier distilled from Llama 3.1 8B Instruct. We annotated all 2 million samples using Llama 3.1 8B Instruct,
in order to get a baseline that we would target with Celadon. The prompt is included in file 2.1, which was created specifically for
target categories within the Common Corpus dataset, but is also displayed below. Should you adapt this process to your own needs, we highly recommend adjusting the
prompt towards categories that occur within your particular dataset. 

```
You are an expert artificial intelligence annotator for toxicity and harmfulness. You are provided the following categories of toxicity:

Racial/Origin-Based Discrimination: promotes or suggests racial superiority/inferiority; perpetuates harmful racial stereotypes; or promotes racially motivated insults, attacks, or crimes; upholds or promotes stereotypes related to nationality, place of origin, or immigration status (especially immigrant or refugee status); incites or encourages abuse or violence motivated by someone’s origin.

Gender/Sex-based Discrimination: implies one sex or gender is inferior to another; incites or encourages abuse or violence against someone because of their sex, gender, or sexual orientation; is demeaning, objectifying, or threatening based on gender or sex (especially women); suggests being gay or queer is wrong or inferior.

Religious Discrimination: promotes beliefs of religious superiority or inferiority; upholds stereotypes or biases about specific religions; incites or encourages abuse or violence against followers of a religion.

Ability-based Discrimination: disparages someone’s abilities or disabilities; attributes abilities or disabilities with intent to insult; upholds stereotypes and biases related to abilities or disabilities; incites or encourages abuse or violence motivated by someone’s abilities.

Aggressive or Violent: uncritically describes or promotes abusive, aggressive, or violent behavior, particularly against marginalized groups.

Annotation problem: Given a piece of text, provide a score between 0 and 3 for each of the proposed categories based on the severity of each. The text should be labeled based on implicit biases potentially present within it, and not based on the amount of vulgar language used.

Rate each text with a score from 0-3, using this scale exactly:
0 - text mentions stereotypes about a particular group of people or acts of violence, but does so in an objective way; text does not promote biases against any group or acts of violence.
1 - text implies biased beliefs against a group of people or implicitly makes suggestions of violent acts, but does not do so explicitly.
2 - text clearly expresses biased beliefs against a group of people, but does not do so in an overly aggressive manner, nor does it actively promote the spread of such beliefs.
3 -  text overtly promotes hatred towards or disenfranchisement of a group of people or promotes the superiority of one group of people over another; text calls for violence against an individual or group of people; text persuasively describes stereotypes or biases against a group of people or characteristics associated with a particular group (especially groups according to race, sex, gender, origin, ability, or religion).

Always respond in this format exactly:

## Racial/Origin-Based Discrimination Score ## :  A score from 0 to 3
## Racial/Origin-Based Discrimination Reasoning ## : What does this text contain that makes it discriminatory based on race or origin? Provide specific examples and detailed reasoning.

## Gender/Sex-Based Discrimination Score ## :  A score from 0 to 3
## Gender/Sex-Based Discrimination Reasoning ## : What does this text contain that makes it discriminatory based on gender or sex? Provide specific examples and detailed reasoning

## Religious Discrimination Score ## :  A score from 0 to 3
## Religious Discrimination Reasoning ## : What does this text contain that makes it religiously discriminatory? Provide specific examples and detailed reasoning.

## Ability-Based Discrimination Score ##:  A score from 0 to 3
## Ability-Based Discrimination Reasoning ##: What does this text contain that makes it discriminatory based on ability? Provide specific examples and detailed reasoning.

## Aggressive or Violent Score ##:  A score from 0 to 3
## Aggressive or Violent Reasoning ##: What does this text contain that creates aggressive or violent themes? Provide specific examples and detailed reasoning.
```
It is worth noting that all definitions and scales were created through intense iteration over multiple versions of the prompt by the PleIAs team, in order to best align the annotations
with our desired outcomes. 

### 3. Combine Data
In order to best utilize the resources available to us, we chose to annotate the data in batches of 10K samples using VLLM. As a result, after annotation, we combined the data back into
one file for each dataset, to make it easier to distribute upon release. You can find the fully annotated dataset [here](https://huggingface.co/datasets/PleIAs/celadon-dataset). 

### 4. Prepare the Data
Despite having a targeted number of samples from each dataset for the initial annotation process, the number of samples is not directly proportional to the number of tokens, which is 
a far more important metric when it comes to representation in the data. As a result, script 4.1 outlines the process which we used to sample the desired number of tokens per language.
We also split the dataset into a train/test set with 70% allocated for training, and 30% allocated for testing, with the test set being split evenly into val/test sets in script 
`4.2_resample.py`. Intermediate script `4.2_prefilter_data.py` was used to clean the dataset of parsing issues that came from the occasional inconsistency of Llama's generated outputs. 

### 5. Model Training
Below are the categorical splits for each dataset (train, val, test) that were used during the training process: 

**Train: 640808 Total Samples**
|             | Race/Origin | Gender/Sex | Religion | Ability | Violence |
|-------------|------------|------------|------------|------------|------------|
| 0s          | 598,244    | 593,870    | 562,894    | 627,438    | 408,575    |
| 1s          | 13,512     | 29,490     | 34,408     | 10,096     | 134,426    |
| 2s          | 18,178     | 16,048     | 34,675     | 2,949      | 68,312     |
| 3s          | 10,874     | 1,400      | 8,831      | 325        | 29,495     |


**Val: 133298 Total Samples**
|             | Race/Origin | Gender/Sex | Religion | Ability | Violence |
|-------------|------------|------------|------------|------------|------------|
| 0s          | 122,610    | 124,277    | 119,722    | 130,774    | 83,158     |
| 1s          | 3,526      | 5,874      | 5,992      | 1,938      | 29,521     |
| 2s          | 4,592      | 2,856      | 6,010      | 520        | 14,457     |
| 3s          | 2,570      | 291        | 1,574      | 66         | 6,162      |

**Test: 133298 Total Samples**
|             | Race/Origin | Gender/Sex | Religion | Ability | Violence |
|-------------|------------|------------|------------|------------|------------|
| 0s          | 122,610    | 124,277    | 119,722    | 130,774    | 83,158     |
| 1s          | 3,526      | 5,874      | 5,992      | 1,938      | 29,521     |
| 2s          | 4,592      | 2,856      | 6,010      | 520        | 14,457     |
| 3s          | 2,570      | 291        | 1,574      | 66         | 6,162      |

In the above tables, any given cell denotes the number of samples in a certain category with a certain classification. A common theme is that ability-based discrimination was by far
the least prevalent within Common Corpus, and as a result this had implications on the training accuracy. 

Celadon was trained using a custom weighted accuracy function in order to account for severe class imbalance between the majority class (samples classified as 0's, or not toxic) and
all minority classes. It was important both to maintain some sort of large gap between the number of benign samples (samples with all 0's), and samples with at least some classified
toxcicity, as well as make sure that we could achieve enough accuracy on those samples which were toxic. As a result, we filtered the data such that the number of rows with no 
toxicity was equal to the number of rows with some toxicity. Further, please refer to our [paper](https://www.arxiv.org/) for a detailed discussion of the weighed loss function used. 

### 6. Evaluation
Celadon was evaluated on the test set shown above, and achieved the following results for each classification head/category:

**Head 0: Race/Origin**

| Metric              | Value   | Confusion Matrix |
|---------------------|---------|-----------------------------|
| **Precision**       | 0.9528  | `[[119789   1441   1056    334]` |
| **Recall**          | 0.9514  | `[   982   2225    283     79]` |
| **F1 Score**        | 0.9520  | `[   948    247   3162    187]` |
| **Accuracy**        | 0.9514  | `[   544    127    253   1641]]` |
| **Balanced Accuracy** | 0.7340  | |
| **Weighted Accuracy** | 0.7340  | |

---

**Head 1: Gender/Sex**

| Metric              | Value   | Confusion Matrix |
|---------------------|---------|-----------------------------|
| **Precision**       | 0.9566  | `[[121480   2169    658     19]` |
| **Recall**          | 0.9549  | `[  1645   3671    409     16]` |
| **F1 Score**        | 0.9557  | `[   600    351   1990     24]` |
| **Accuracy**        | 0.9549  | `[    29     30     56    151]]` |
| **Balanced Accuracy** | 0.7138  | |
| **Weighted Accuracy** | 0.7138  | |

---

**Head 2: Religion**

| Metric              | Value   | Confusion Matrix |
|---------------------|---------|-----------------------------|
| **Precision**       | 0.9399  | `[[115125   3033   1498    177]` |
| **Recall**          | 0.9310  | `[  1239   3618    890     79]` |
| **F1 Score**        | 0.9348  | `[   670    751   4380    228]` |
| **Accuracy**        | 0.9310  | `[   199    128    302    981]]` |
| **Balanced Accuracy** | 0.7294  | |
| **Weighted Accuracy** | 0.7294  | |

---

**Head 3: Ability**

| Metric              | Value   | Confusion Matrix |
|---------------------|---------|-----------------------------|
| **Precision**       | 0.9845  | `[[129739    751    122      5]` |
| **Recall**          | 0.9849  | `[   812   1173     58      1]` |
| **F1 Score**        | 0.9847  | `[   201     36    323      1]` |
| **Accuracy**        | 0.9849  | `[    18      5      4     49]]` |
| **Balanced Accuracy** | 0.6969  | |
| **Weighted Accuracy** | 0.6969  | |

---

**Head 4: Violence**

| Metric              | Value   | Confusion Matrix |
|---------------------|---------|-----------------------------|
| **Precision**       | 0.8186  | `[[70466 10865  1881   276]` |
| **Recall**          | 0.7992  | `[ 4072 21710  3040   491]` |
| **F1 Score**        | 0.8058  | `[  774  2612 10144   849]` |
| **Accuracy**        | 0.7992  | `[  248   616  1042  4212]]` |
| **Balanced Accuracy** | 0.7446  | |
| **Weighted Accuracy** | 0.7446  | |

