<picture>
  <source media="(prefers-color-scheme: dark)" srcset="./images/pleias%20logo%20(white).svg">
  <source media="(prefers-color-scheme: light)" srcset="./images/pleias%20logo%20(black).svg">
  <img alt="PleIAs Logo" src="./images/pleias%20logo%20(black).svg">
</picture>

<h1 align="center">Toxic Commons</h1>


![License](https://img.shields.io/badge/license-MIT-blue) ![HuggingFace](https://img.shields.io/badge/🤗%20Hugging%20Face-Model-blue) ![HuggingFace](https://img.shields.io/badge/🤗%20Hugging%20Face-Dataset-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-black?logo=PyTorch) 

The official repository of Toxic Commons, a framework for filtering LLM pretraining data created by [PleIAs](https://huggingface.co/PleIAs)
Toxic Commons and the related model, Celadon, are the result of the paper [Toxicity of the Commons: Curating Open Source Pre-Training Data](https://www.arxiv.org) by Catherine Arnett and Eliot Jones. 
An internal version of Celadon was applied to the PleIAs family of models prior to pretraining, in order to reduce the possibility of harmful behaviors and biases. 
Celadon was created to be a more efficient classifier of toxic data, in order to reserve compute and time for larger model training. In its current form, Celadon
is designed to work with [Common Corpus](https://huggingface.co/collections/PleIAs/common-corpus-65d46e3ea3980fdcd66a5613), the largest public domain dataset
for training LLMs, and as a result may not work out of the box on alternative datasets (such as those with mainly webtext). In releasing this repository, we hope that
others will adapt this methodology to better filter their pretraining data in the future.   

## Table of Contents
- [Pipeline Overview](#pipeline)
- [Installation](#installation)
- [Usage](#usage)
- [Citation](#citation)
- [License](#license)

## Pipeline
The Celadon model is the first step in a three-step pipeline outlined in our paper, Toxicity of the Commons, which can be used to synthetically realign harmful pretraining
data. It was designed to be used as follows:

:sparkles:**STEP ONE:**:sparkles: For data that needs to be filtered, run the data through Celadon to obtain an approximation for the toxicity level of the data, as well as what axes it is
toxic on. Celadon trades off pure accuracy for performance, which allows it to obtain toxicity determinations that are in the ballpark of Llama 3.1 8B Instruct, while
being far more efficient. The purpose of the multi-step pipeline is to provide additional checks on Celadon, in order to better align the data to less harmful standards. 

:sparkles:**STEP TWO:** :sparkles: We carefully set toxicity thresholds to further separate the samples in our dataset into three different categories:
1.  *No Toxicity*: A total score of 0-3 with no individual category score exceeding a 2. These texts either have no toxic content, or very low levels
  not warranting any further action. We allow some breathing room with scores of 2 for any overly-sensitive cases.
2. *Mild Toxicity*: A total score of between 4 and 6, or a total score of 3 where that score comes from a 3 in a single category. For a lower score of 3,
   we require the sole contribution to come from a single category, while higher scores indicate toxicity across multiple categories in a significant manner.
3. *Toxic Content*: The sum of scores is 7 or greater. This text likely contains toxic content that requires scrutiny from an LLM annotator.

:sparkles:**STEP THREE:** :sparkles: We bring LLM annotators (Llama 3.1 8B Instruct) back into the loop in step three (after being used initially to annotate the data that Celadon was
trained on, though if you use Celadon yourself, this is the first instance of using an LLM annotator). Samples which were labeled as *not toxic* are left in the pretraining dataset, 
while samples labeled *mildly toxic* or *toxic* are removed from the pretraining set, and will instead be reintroduced during the annealing phase. Doing so allows us to still maintain 
the benefits that could come from including these samples in model training, while also giving us the opportunity to re-align the content with less harmful standards. The LLM annotator
is used to generate content warnings for samples labeled *mildly toxic*, and it is used to synthetically rewrite the *toxic* content. In either case, if the LLM annotator deems that the
sample has been misclassified, it is instructed not to change the content, but instead provide justification. This justification makes its way into the annealing corpus, in order to 
better align the model with our preferences (or, in this case, the preferences of a previously RLHFed instruct model). Prompts are for this step are included under the `/prompts` dir. 

## Installation 
Celadon is compatible with HuggingFace Transformers, with source code located [here](https://huggingface.co/PleIAs/celadon). To load the model, all you have to do is:

1. Download the repository: `git clone https://huggingface.co/PleIAs/celadon`
2. Import the model class `MultiHeadDebertaForSequenceClassification` from `model.py`: `from celadon.model import MultiHeadDebertaForSequenceClassification`
3. Import AutoTokenizer from `transformers`: `from transformers import AutoTokenizer`
4. Load the tokenizer: `tokenizer = AutoTokenizer.from_pretrained("celadon")`
5. Load the model: `model = MultiHeadDebertaForSequenceClassification.from_pretrained("celadon")`

## Usage
Here is a sample script that classifies a single sample using Celadon:
```
from transformers import AutoTokenizer
from celadon.model import MultiHeadDebertaForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("celadon")
model = MultiHeadDebertaForSequenceClassification.from_pretrained("celadon")
model.eval()

sample_text = "This is an example of a normal sentence"

inputs = tokenizer(sample_text, return_tensors="pt", padding=True, truncation=True)
outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])

categories = ['Race/Origin', 'Gender/Sex', 'Religion', 'Ability', 'Violence']
predictions = outputs.argmax(dim=-1).squeeze().tolist()

# Print the classification results for each category
print(f"Text: {sample_text}")
for i, category in enumerate(categories):
    print(f"Prediction for Category {category}: {predictions[i]}")
```
Note: Celadon was written in PyTorch – though we have tried to make it as accessible as possible to use for everyone by adapting it to Transformers, 
those more familiar with `torch` may prefer to use it that way. 

## Citation
```
@article{arnett2024toxicity,
  title={{Toxicity of the Commons: Curating Open-Source Pre-Training Data}},
  author={Arnett, Catherine and Jones, Eliot and Yamshchikov, Ivan P. and Langlais, Pierre-Carl},
  journal={arXiv preprint arXiv:2410.22587},
  url={https://arxiv.org/pdf/2410.22587},
  year={2024}
}
```

## License
```
MIT License

Copyright (c) 2024 pleias

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

