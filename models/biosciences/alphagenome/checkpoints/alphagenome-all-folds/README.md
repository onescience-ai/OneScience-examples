---
license: other
license_name: alphagenome
license_link: https://deepmind.google.com/science/alphagenome/model-terms
extra_gated_heading: Access AlphaGenome on Hugging Face
extra_gated_prompt: >-
  AlphaGenome is provided for non-commercial use only and is subject to the
  [Model Terms of Use](https://deepmind.google.com/science/alphagenome/model-terms).
  To accept terms, please login, complete the required fields and click Accept.
  Requests are processed immediately.
extra_gated_button_content: Accept and continue
extra_gated_fields:
  Organization, university, or other affiliation(s): text
language:
- en
tags:
- biology
---

## Description

AlphaGenome is a unified DNA sequence model designed to advance regulatory
variant-effect prediction and shed light on genome function. It analyzes DNA
sequences of up to 1 million base pairs to deliver predictions at single
base-pair resolution across diverse modalities, including gene expression,
splicing patterns, chromatin features, and contact maps.

The core architecture features a U-Net-style design that combines an encoder for
downsampling, transformers with inter-device communication to capture long-range
interactions, and a decoder for upsampling. These components feed into
task-specific output heads that generate predictions at their respective
assay-specific resolutions.

By achieving state-of-the-art performance across diverse genomic benchmarks,
AlphaGenome provides a robust framework for understanding the molecular function
of DNA sequences and interpreting non-coding variation.

## Inputs and outputs

*   **Input:** Up to 1 Mb (2\*\*20) one-hot encoded DNA sequence, and an
    organism type index (representing human or mouse).
*   **Output:** 11 diverse modalities, including:
    *   RNA expression (RNA-Seq, CAGE-seq and PRO-cap).
    *   Chromatin accessibility (DNase-seq and ATAC-seq).
    *   Histone modifications.
    *   Transcription factor binding.
    *   Chromatin contact maps.
    *   Splice sites and their usage.
    *   Splice junction coordinates and strength.

## Citation

```
@article{alphagenome,
  title={Advancing regulatory variant effect prediction with {AlphaGenome}},
  author={Avsec, {\v Z}iga and Latysheva, Natasha and Cheng, Jun and Novati, Guido and Taylor, Kyle R. and Ward, Tom and Bycroft, Clare and Nicolaisen, Lauren and Arvaniti, Eirini and Pan, Joshua and Thomas, Raina and Dutordoir, Vincent and Perino, Matteo and De, Soham and Karollus, Alexander and Gayoso, Adam and Sargeant, Toby and Mottram, Anne and Wong, Lai Hong and Drot{\'a}r, Pavol and Kosiorek, Adam and Senior, Andrew and Tanburn, Richard and Applebaum, Taylor and Basu, Souradeep and Hassabis, Demis and Kohli, Pushmeet},
  journal={Nature},
  volume={649},
  number={8099},
  year={2026},
  doi={10.1038/s41586-025-10014-0},
  publisher={Nature Publishing Group UK London}
}
```

## Installation

To install the accompanying code necessary to run the model, please run the
following:

```shell
$ pip install git+https://github.com/google-deepmind/alphagenome_research.git
```

## Usage

In addition to the model, we provide a DNA model class that wraps the core model
and provides a more intuitive set of functions for creating predictions, scoring
variants, performing in silico mutagenesis (ISM) and more.

Here's an example of making a variant prediction:

```python
from alphagenome.data import genome
from alphagenome.visualization import plot_components
from alphagenome_research.model import dna_model
import matplotlib.pyplot as plt

model = dna_model.create_from_huggingface('all_folds')

interval = genome.Interval(chromosome='chr22', start=35677410, end=36725986)
variant = genome.Variant(
    chromosome='chr22',
    position=36201698,
    reference_bases='A',
    alternate_bases='C',
)

outputs = model.predict_variant(
    interval=interval,
    variant=variant,
    ontology_terms=['UBERON:0001157'],
    requested_outputs=[dna_model.OutputType.RNA_SEQ],
)

plot_components.plot(
    [
        plot_components.OverlaidTracks(
            tdata={
                'REF': outputs.reference.rna_seq,
                'ALT': outputs.alternate.rna_seq,
            },
            colors={'REF': 'dimgrey', 'ALT': 'red'},
        ),
    ],
    interval=outputs.reference.rna_seq.interval.resize(2**15),
    # Annotate the location of the variant as a vertical line.
    annotations=[plot_components.VariantAnnotation([variant], alpha=0.8)],
)
plt.show()
```

## Model Data

AlphaGenome was trained to predict read coverage for a wide range of different
functional genomics assays – including RNA-seq, DNase-seq, CAGE, ChIP-seq
–&nbsp;directly from human and mouse reference genome sequences. The training
data was sourced from large public consortia including
[ENCODE](http://encodeproject.org/), [GTEx](https://www.gtexportal.org/),
[4D Nucleome](https://4dnucleome.org/) and
[FANTOM5](https://fantom.gsc.riken.jp/5/), encompassing experimental
measurements of key regulatory modalities across hundreds of cell types and
tissues. For more information, please refer to the
[AlphaGenome manuscript](https://www.biorxiv.org/content/10.1101/2025.06.25.661532v2).

Along with the release of the weights and model inference code, we are releasing
the complete training, validation and test datasets used for AlphaGenome. The
data is stored in compressed TFRecord format and can be loaded as follows:

```python
from alphagenome.data import fold_intervals
from alphagenome_research.io import dataset
from alphagenome_research.model import dna_model

ds_iter = dataset.create_dataset(
    organism=dna_model.Organism.HOMO_SAPIENS,
    fold_split=dna_model.ModelVersion.ALL_FOLDS,
    subset=fold_intervals.Subset.TRAIN,
).as_numpy_iterator()
element = next(ds_iter)
```

Each element in the dataset is a tuple, where each item corresponds to a
specific data bundle. The example below illustrates this structure, showing the
ATAC and DNase bundles alongside the input DNA sequence, masks, and interval
metadata:

```python
({"atac": "bfloat16[1052672,256]",
  "atac_mask": "bool[1,256]",
  "dna_sequence": "float32[1052672,4]",
  "interval/chromosome": b"chr7",
  "interval/end": "int64[]",
  "interval/start": "int64[]"},
 {"dna_sequence": "float32[1052672,4]",
  "dnase": "bfloat16[1052672,384]",
  "dnase_mask": "bool[1,384]",
  "interval/chromosome": b"chr7",
  "interval/end": "int64[]",
  "interval/start": "int64[]"},
...
)
```

**Notes:**

*   **Interval extension**: To support data augmentation via sequence shifting
    during training, the 1 Mb intervals were extended by 4,096 base pairs (2,048
    bp on each side).
*   **GTEx data exclusion**: The released dataset does not contain GTEx tissue
    data due to licensing restrictions, though the corresponding column headers
    remain in the bundle’s metadata. However, users can easily extend the
    dataset with GTEx or other external sources using the provided interval
    metadata (chromosome, start, end).

## Implementation Information

### Hardware

AlphaGenome training involved a two-stage process: pre-training and
distillation. Pre-training was carried out on 256
[Tensor Processing Units (TPUv3)](https://docs.cloud.google.com/tpu/docs/v3)
using sequence parallelism across groups of 4 interconnected chips. We leverage
TPU pods (large clusters of TPUs) to provide a scalable solution for training,
to enable large batch sizes which can lead to better model quality. For
distillation, AlphaGenome was trained on 64 NVIDIA H100 GPUs without sequence
parallelism. The evaluation of all models was carried out on NVIDIA H100 GPUs
without sequence parallelism.

### Software

Training was done using [JAX](https://github.com/google/jax) and
[JAXline](https://github.com/google-deepmind/jaxline).

JAX allows researchers to take advantage of the latest generation of hardware,
including TPUs, for faster and more efficient training of large models.

JAXline is a distributed JAX training and evaluation framework. It is designed
to be forked, covering only the most general aspects of experiment boilerplate.

## Evaluation

AlphaGenome demonstrates state-of-the-art performance across a diverse set of
genomic prediction tasks. The model was evaluated using two primary approaches:

1.  **Genome Track Prediction**: Assessing the ability to predict functional
    genomic signals (read coverage) on previously unseen DNA sequences (held-out
    test intervals).
2.  **Variant Effect Prediction (VEP):** Assessing the ability to predict the
    molecular consequences of genetic variants (e.g., single nucleotide
    variants) by comparing predictions for reference and alternative alleles
    against ground-truth datasets (e.g., experimental QTL effect sizes, readouts
    from reporter assays).

Key Highlights:

*   **Broad SOTA Performance:** AlphaGenome matched or outperformed the best
    available external models on **22 out of 24** genome track prediction
    evaluations.
*   **Variant Interpretation:** For variant effect prediction, AlphaGenome
    matched or exceeded top-performing external models on **25 out of 26**
    evaluations.
*   **Multimodal Capability:** Unlike specialized models, AlphaGenome jointly
    predicts all assessed modalities –&nbsp;including splicing, expression,
    accessibility, and 3D contact maps –&nbsp;within a single framework.
*   **Single-Pass Efficiency:** The distilled student model achieves this
    performance and broad coverage with a single inference pass, eliminating the
    need for complex model ensembling.

The tables below detail the performance metrics for specific modalities and
tasks.

## Benchmark Results

The following table focuses on the accuracy of the pre-trained, non-distilled
model in predicting genomic tracks on unseen sequences:

Modality              | Evaluation                       | Metric      | Resolution | Value | Baseline Model | Relative Improvement (%)
:-------------------- | :------------------------------- | :---------- | :--------- | :---- | :------------- | :-----------------------
**Splicing**          | Splice site classification       | auPRC       | 1 bp       | 0.79  | DeltaSplice    | 1.0
&nbsp;                | Splice site usage                | Pearson r   | 1 bp       | 0.86  | DeltaSplice    | 6.7
**RNA expression**    | RNA-seq coverage                 | Pearson r   | 1 bp       | 0.59  | Borzoi         | 28.2
&nbsp;                | RNA-seq coverage                 | Pearson r   | 32 bp      | 0.78  | Borzoi         | 4.6
&nbsp;                | RNA-seq gene expr. LFC           | Pearson r   | Gene       | 0.57  | Borzoi         | 14.7
&nbsp;                | CAGE coverage                    | Pearson r   | 32 bp      | 0.74  | Borzoi         | 4.4
&nbsp;                | CAGE coverage                    | Pearson r   | 128 bp     | 0.71  | Enformer       | \-0.3
&nbsp;                | Alternative PA                   | Spearman r  | Gene       | 0.87  | Borzoi         | 13.1
**DNA accessibility** | DNase-seq coverage               | Profile JSD | 1 bp       | 0.51  | ChromBPNet     | 6.4
&nbsp;                | DNase-seq coverage               | Pearson r   | 32 bp      | 0.86  | Borzoi         | 4.7
&nbsp;                | DNase-seq coverage               | Pearson r   | 128 bp     | 0.87  | Enformer       | 2.4
&nbsp;                | ATAC-seq coverage                | Profile JSD | 1 bp       | 0.46  | ChromBPNet     | 1.6
&nbsp;                | ATAC-seq coverage                | Pearson r   | 32 bp      | 0.57  | Borzoi         | 3.4
&nbsp;                | ATAC-seq coverage                | Pearson r   | 128 bp     | 0.72  | Enformer       | 2.7
**Histone mods**      | Histone ChIP-seq coverage        | Pearson r   | 32 bp      | 0.69  | Borzoi         | 3.1
&nbsp;                | Histone ChIP-seq coverage        | Pearson r   | 128 bp     | 0.71  | Enformer       | 2.4
**TF binding**        | TF ChIP-seq coverage             | Pearson r   | 32 bp      | 0.55  | Borzoi         | 5.0
&nbsp;                | TF ChIP-seq coverage             | Pearson r   | 128 bp     | 0.58  | Enformer       | 1.4
**DNA contact maps**  | Orca Contact maps                | Pearson r   | 4000 bp    | 0.79  | Orca           | 6.3
&nbsp;                | Orca Contact maps cell type diff | Pearson r   | 4000 bp    | 0.42  | Orca           | 42.3

And the next table details the performance of the distilled model in predicting
the functional effects of genetic variants:

Modality              | Evaluation                    | Type      | Metric     | Value | Baseline Model | Relative Improvement (%)
:-------------------- | :---------------------------- | :-------- | :--------- | :---- | :------------- | :-----------------------
**Splicing**          | ClinVar splice site region    | Causality | auPRC      | 0.57  | Pangolin       | 3.7
&nbsp;                | ClinVar noncoding             | \-        | auPRC      | 0.66  | Pangolin       | 2.9
&nbsp;                | ClinVar missense              | \-        | auPRC      | 0.18  | DeltaSplice    | 13.7
&nbsp;                | Splicing outlier (zero-shot)  | \-        | auPRC      | 0.22  | Pangolin       | 59.1
&nbsp;                | Splicing outlier (supervised) | \-        | auPRC      | 0.28  | AbSplice       | 13.0
&nbsp;                | sQTL                          | Causality | auPRC      | 0.76  | Pangolin       | 13.9
&nbsp;                | MFASS                         | \-        | auPRC      | 0.51  | Pangolin       | \-5.7
**RNA expression**    | eQTL                          | Direction | Spearman r | 0.49  | Borzoi         | 25.5
&nbsp;                | eQTL (zero-shot)              | Causality | auROC      | 0.71  | Borzoi         | 5.4
&nbsp;                | eQTL (supervised)             | Causality | auROC      | 0.80  | Borzoi         | 15.6
&nbsp;                | ENCODE E2G (zero-shot)        | \-        | auPRC      | 0.75  | Borzoi         | 13.0
&nbsp;                | paQTL                         | \-        | auPRC      | 0.63  | Borzoi         | 7.3
**DNA accessibility** | CAGI5 MPRA                    | Causality | Pearson r  | 0.65  | Borzoi         | 6.3
&nbsp;                | ds/caQTL                      | Direction | Pearson r  | 0.70  | ChromBPNet     | 7.7
&nbsp;                | ds/caQTL                      | Causality | auPRC      | 0.52  | Borzoi         | 18.0
**TF binding**        | bQTL                          | Direction | Pearson r  | 0.55  | Borzoi         | 2.8
&nbsp;                | bQTL                          | Causality | auPRC      | 0.50  | Borzoi         | 6.0

## Usage and Limitations

### License

Unless required by applicable law or agreed to in writing, all software and
materials distributed here (under the
[Apache 2.0 License](https://github.com/google-deepmind/alphagenome_research/blob/main/README.md)
with respect to model code and
[non-commercial terms](https://deepmind.google.com/science/alphagenome/model-terms)
for the model parameters) are distributed on an "AS IS" BASIS, WITHOUT
WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
licenses for the specific language governing permissions and limitations under
those licenses.

### Intended usage

*   Non-Commercial use only: The model parameters are restricted to
    non-commercial use by non-commercial organizations (e.g., universities,
    non-profits, research institutes, and journalism). It must not be used for
    any commercial activities or on behalf of commercial entities.
*   Model derivatives: If you fine-tune or modify this model, the resulting
    model is classified as a "Derivative". All derivatives are subject to these
    exact same terms, meaning strictly no commercial use is permitted for
    fine-tuned versions.
*   Distillation: Training a new model using the outputs or predictions of
    AlphaGenome is also restricted; resulting models must be governed by the
    AlphaGenome Model Parameters Terms of Use

### Limitations

*   Like other sequence-based models, accurately capturing the influence of very
    distant regulatory elements, like those over 100,000 DNA letters away, is
    still an ongoing challenge. Another priority for future work is further
    increasing the model’s ability to capture cell- and tissue-specific
    patterns. We haven't designed or validated AlphaGenome for personal genome
    prediction, a known challenge for AI models. Instead, we focused more on
    characterising the performance on individual genetic variants. And while
    AlphaGenome can predict molecular outcomes, it doesn't give the full picture
    of how genetic variations lead to complex traits or diseases. These often
    involve broader biological processes, like developmental and environmental
    factors, that are beyond the direct scope of our model.
