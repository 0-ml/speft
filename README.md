# **Refining Salience-Aware Sparse Fine-Tuning Strategies for Language Models**

---

This is the implementation of ACL 2025 paper： [Refining Salience-Aware Sparse Fine-Tuning Strategies for Language Models](https://arxiv.org/abs/2412.13488)

### Requirements

---

To run our code, please install all the dependency packages by using the following command:

    `pip install -r requirements.txt`

If you find any libraries not listed in `requirements.txt` are needed when running the code, **please let us know** so we can update the file promptly. Thank you!

### Experiments

---

The datasets used by the Python files in the `train` folder for fine-tuning and evaluation are shown in the following table respectively:

| Files       | Fine-tune     | Eval           |
| ----------- | ------------- | -------------- |
| run_glue.py | GLUE          | GLUE           |
| run_mmlu.py | Alpaca/OASST2 | MMLU           |
| run_math.py | MetaMathQA    | GSM8K          |
| run_code.py | CodeFeedback  | HumanEval/MBPP |

### Baselines

---

We provide LoRA and PiSSA as baselines. You can apply them by running the scripts in `scripts/lora` and `scripts/pissa`.

### SPEFT

---

You can apply SPEFT by running the scripts in `scripts/sparse`.

We explain some of the arguments as follows:

* `sparse_method:` The salience metric used to compute scores and construct sparse masks
* `sparse_global_topk:` If True, global sparsity masking will be used rather than local sparsity masking
* `sparse_k:` During fine-tuning, the model selects weights ranked in the top-k by scores as trainable parameters. This can be expressed either as **absolute counts** (e.g., 25,559,040) or **percentage of total parameters** (e.g., 0.0098).
* `sparse_mask_update_steps:` Steps interval for dynamic masks update. If set to 0, uses static masks (no updates during fine-tuning)

### Bugs or questions

---

If you have any questions related to the codes or the paper, please contact Xinxin Liu ( `xx.liu@siat.ac.cn` ) or open an issue.
