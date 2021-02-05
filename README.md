# dorothea-py
This is a Python implementation of the package [DoRothEA](https://github.com/saezlab/dorothea) for Transcript Factor activity prediction from transcriptomics data.
In this implementation, DoRothEA is coupled with the statistical method [SCIRA](https://github.com/WangNing0420/SCIRA) and uses the [scanpy](https://github.com/theislab/scanpy) framework. 


## Installation
The package can easly be installed using pip
```
pip install git+https://github.com/saezlab/dorothea-py.git
```

## Citing DoRothEA
Beside the original paper there are two additional papers expanding the usage of DoRothEA regulons.

* If you use DoRothEA for your research please cite the original publication: 
> Garcia-Alonso L, Holland CH, Ibrahim MM, Turei D, Saez-Rodriguez J. "Benchmark and integration of resources for the estimation of human transcription factor activities." _Genome Research._ 2019. DOI: [10.1101/gr.240663.118](https://doi.org/10.1101/gr.240663.118).

* If you use the mouse version of DoRothEA please cite additionally:
> Holland CH, Szalai B, Saez-Rodriguez J. "Transfer of regulatory knowledge from human to mouse for functional genomics analysis." _Biochimica et Biophysica Acta (BBA) - Gene Regulatory Mechanisms._ 2019. DOI: [10.1016/j.bbagrm.2019.194431](https://doi.org/10.1016/j.bbagrm.2019.194431).

* If you apply DoRothEA's regulons on single-cell RNA-seq data please cite additionally:
> Holland CH, Tanevski J, Perales-Patón J, Gleixner J, Kumar MP, Mereu E, Joughin BA, Stegle O, Lauffenburger DA, Heyn H, Szalai B, Saez-Rodriguez, J. "Robustness and applicability of transcription factor and pathway analysis tools on single-cell RNA-seq data." _Genome Biology._ 2020. DOI: [10.1186/s13059-020-1949-z](https://doi.org/10.1186/s13059-020-1949-z).


## Usage of DoRothEA for commercial purpose
DoRothEA as it stands is intended only for academic use as in contains resources whose licenses don't permit commerical use. However, we developed a non-academic version of DoRothEA by removing those resources (mainly TRED from the curated databases). You find the non-academic package with the regulons [here](https://github.com/saezlab/dorothea/tree/non-academic).

## History
Initially DoRothEA was a framework of **D**iscriminant **R**egulon **E**xpression **A**nalysis, as described in [Garcia-Alonso et al., 2017](https://cancerres.aacrjournals.org/content/early/2017/12/09/0008-5472.CAN-17-1679). At that time DoRothEA was used with a smaller set of regulons to find associations between TF activities and drug responses. Those results are queryable at [Open Targets](https://dorothea.opentargets.io/). Over the past years DoRothEA evolved from the analytical framework to a pure regulon resource as it is now.