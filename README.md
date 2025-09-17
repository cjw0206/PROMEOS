# PROMEOS

**PROMEOS** is a multimodal learning framework for protein-protein interaction prediction, leveraging a Mixture of Experts to integrate ontology and sequence embeddings.


## Model Architecture

<p align="center">
  <img src="Overall_Workflow.jpg" alt="PROMEOS Architecture" width="600"/>
</p>

<p align="left">
  <b>(a)</b> GO and sequence embeddings from a protein pair are concatenated and encoded through a Transformer encoder with sparse MoE layers.  
  The encoded representations are element-wise multiplied, followed by a weighted attention and a linear layer for final prediction.<br>
  <b>(b)</b> GO embeddings are derived from Node2Vec trained on a random walk corpus over the GO graph.  
  Red symbols next to GO terms indicate proteins annotated with those terms (e.g., $P_1$, $P_2$).<br>
  <b>(c)</b> Protein sequences are tokenized, truncated, and encoded using ESM-2; the resulting embeddings are reshaped to match the GO embedding dimension.
</p>

## ⚙Requirements

- **Python**: 3.9  
- **PyTorch**: 2.1.2 (with CUDA 12.1 support)  
- **torchvision**: 0.16.2  
- **torchaudio**: 2.1.2  
- **transformers**: 4.51.3  
- **tokenizers**: 0.21.1  
- **numpy**: 1.26.3  
- **pandas**: 2.2.3  
- **scikit-learn**: 1.6.1  
- **scipy**: 1.13.1  
- **matplotlib**: 3.9.4  
- **seaborn** (optional, if used for plotting)  
- **tqdm**: 4.67.1  


