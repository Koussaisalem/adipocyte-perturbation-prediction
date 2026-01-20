# Method Description

Our method uses a Graph-Augmented Flow Matching model to predict cellular responses to gene perturbations.
The model combines three components: a GATv2 graph neural network encoder, a conditional flow matching decoder, and a proportion prediction head.
The GATv2 encoder maps gene perturbations to 256-dimensional embeddings using a biological knowledge graph containing 11,014 genes and 51,450 regulatory edges.
The flow matching decoder generates 100 synthetic single-cell expression profiles for each perturbation by solving an ordinary differential equation that transforms Gaussian noise into realistic cell states.
The proportion prediction head estimates the distribution of cells across four adipocyte programs: pre-adipocyte, adipocyte, lipogenic, and other states.
The model operates in PCA space (300 components) for computational efficiency and transforms predictions back to full gene expression space.
During inference, we start from a baseline non-committed cell state and apply the perturbation embedding to generate the predicted cell distribution.
This approach enables zero-shot prediction for any gene in the knowledge graph, even those without direct experimental observations.
The model was trained using a combination of conditional flow matching loss, maximum mean discrepancy, and proportion loss to ensure both expression and state distribution accuracy.

# Rationale

The key challenge in perturbation prediction is generalizing to unseen genes without requiring experimental measurements for every possible perturbation.
We address this through graph-based transfer learning, where genes with similar regulatory roles in the knowledge graph produce similar perturbation effects.
Flow matching was chosen over traditional regression approaches because cellular responses are inherently stochastic and multi-modal, requiring distributional rather than point predictions.
The knowledge graph integrates three complementary data sources: transcription factor regulation from CollecTRI and DoRothEA, protein-protein interactions from STRING, and functional relationships from Gene Ontology.
This multi-scale biological prior enables the model to reason about gene function through network context, regulatory pathways, and known biological processes.
The GATv2 architecture with multi-hop message passing allows information to propagate through the graph, capturing indirect relationships and pathway-level effects.
Generating 100 cells per perturbation provides a robust estimate of the expression distribution and enables calculation of program proportions.
Early stopping on validation MMD (0.0129) ensured the model learned generalizable patterns rather than memorizing training perturbations.
The proportion head provides an interpretable summary of cellular state changes that complements the full expression predictions.

# Data and Resources Used

The training data consists of 44,846 single cells from an adipocyte differentiation time course, provided by the competition organizers.
We used approximately 100 genes with observed perturbation data for model training and 5 held-out genes for validation.
The knowledge graph was constructed from three public databases: CollecTRI and DoRothEA for transcription factor regulatory networks, STRING for protein-protein interactions, and Gene Ontology for functional annotations.
Gene embeddings were extracted from the pretrained Geneformer model, a transformer-based foundation model trained on 30 million single-cell transcriptomes.
Principal component analysis with 300 components was applied to the expression data for dimensionality reduction while retaining 95 percent of variance.
The model was implemented in PyTorch version 2.0 with PyTorch Geometric for graph operations and torchdiffeq for ODE integration.
Training was performed on a single NVIDIA GPU with 16GB memory for approximately 1 epoch using AdamW optimizer with learning rate 1e-4.
All preprocessing and model training code was developed specifically for this competition using standard scientific Python libraries including scanpy, pandas, and numpy.
No external single-cell perturbation datasets were used beyond the provided training data and public knowledge graphs.
