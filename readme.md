# Machine Translation Hallucination Detection for Low and High Resource Languages using Large Language Models

## Overview

This repository presents our work for the paper **"Machine Translation Hallucination Detection for Low and High Resource Languages using Large Language Models"**. The paper evaluates hallucination detection approaches using Large Language Models (LLMs) and semantic similarity within massively multilingual embeddings, addressing a critical challenge in machine translation. 
Code will be added later.

## Abstract

Recent advancements in massively multilingual machine translation systems have significantly enhanced translation accuracy; however, even the best performing systems still generate hallucinations, severely impacting user trust. Detecting hallucinations in Machine Translation (MT) remains a critical challenge, particularly since existing methods excel with High-Resource Languages (HRLs) but exhibit substantial limitations when applied to Low-Resource Languages (LRLs). This study spans 16 language directions, covering HRLs and LRLs with diverse scripts, and finds that the choice of model is essential for performance.

## Key Findings

- For HRLs, Llama3-70B outperforms the previous state of the art by as much as 0.16 MCC (Matthews Correlation Coefficient).
  
- For LRLs, Claude Sonnet shows superior performance on average by 0.03 MCC compared to other LLMs.

- LLMs can achieve performance comparable or even better than previously proposed models despite not being explicitly trained for any machine translation task, though their advantage is less significant for LRLs.


## Citation

If you use this repository or refer to our work, please cite our paper as follows:

```
@article{gongas2024machine,
  title={Machine Translation Hallucination Detection for Low and High Resource Languages using Large Language Models},
  author={Laura Gongas et al.},
  journal={arXiv preprint arXiv:2407.16470},
  year={2024}
}
```

## Contact

For further inquiries or collaboration opportunities, please contact us at [kenza.benkirane.23@ucl.ac.uk].
