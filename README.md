# PROMEOS

**PROMEOS** is a multimodal learning framework for protein-protein interaction prediction, leveraging a Mixture of Experts to integrate ontology and sequence embeddings.


## Model Architecture

<p align="center">
  <img src="Overall_Workflow.jpg" alt="PROMEOS Architecture" width="600"/>
</p>

<p align="center">
  <b>(a)</b> GO and sequence embeddings from a protein pair are concatenated and encoded through a Transformer encoder with sparse MoE layers.  
  The encoded representations are element-wise multiplied, followed by a weighted attention and a linear layer for final prediction.  
  <b>(b)</b> GO embeddings are derived from Node2Vec trained on a random walk corpus over the GO graph.  
  Red symbols next to GO terms indicate proteins annotated with those terms (e.g., $P_1$, $P_2$).  
  <b>(c)</b> Protein sequences are tokenized, truncated, and encoded using ESM-2; the resulting embeddings are reshaped to match the GO embedding dimension.
</p>

## About
PROMEOS applies [your method/approach] to address [research problem].  
This repository provides:
- Data preprocessing
- Model training and evaluation
- End-to-end reproducibility of results

---

## Installation
Clone this repository and install the required dependencies:

```bash
git clone https://github.com/<your-username>/PROMEOS.git
cd PROMEOS
pip install -r requirements.txt
