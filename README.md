# MoR Implementation

## Install
The installation is for Linux

Install dependencies
   ```bash
   conda create -n mor python=3.10 -y
   conda activate mola
   pip install -r requirements.txt
   ```
### Data Preparation Scripts
We take ScienceQA as an example, which can be applied to customized datasets as well.   
```bash
python preparation_scienceqa_data.py \
         --save_path "./scienceqa"
```
This script takes the HuggingFace dataset as input and processes the data sample following the format and code at  **[mm-cot](https://github.com/amazon-science/mm-cot)**.

Note: We use the **[HuggingFace datasets](https://huggingface.co/docs/datasets/en/index)** library to load the ScienceQA dataset from HuggingFace Hub.
If you want to load datasets from local files or other methods, please refer to this **[tutorial](https://huggingface.co/docs/datasets/en/loading)** to modify the script accordingly.
In most cases, you only need to modify line 110 as in the preparation_scienceqa_data.py:
```bash
datasets_scienceqa = load_dataset(args.dataset) # Modify this line to load datasets in different format
```

The processed data sample should contain four essential components `instruction`, `input`, and `output` are used for downstream task instruction tuning. `answer` is used in evaluation only.

| <!-- -->    | <!-- -->    |
| --- | --- |
| instruction | The question and the choices. (`Question: ...? Options: (A) ... (B) ...`) |
| input | Not used in this situation.|
| output | The answer (`Answer: The answer is B.`) |
| answer | The answer to the question which is used for evaluation. (`B`)|

The output of the scripts contains one Huggingface dataset (`science_qa.hf`) for training and three JSON files. The `scienceq_test.json` is for the evaluation.  `scienceq_train.json` and `scienceq_validation.json` are not necessary.
```
├── scienceqa
│   └── science_qa.hf
│   └── scienceq_train.json
│   └── scienceq_validation.json
│   └── scienceq_test.json
```

### Training (`mola_training.py`)
The input data for the training script is the `science_qa.hf` that we processed in the previous step.
We briefly introduce the important hyperparameters as follows: 

| Hyperparameters    | <!-- -->    |
| --- | --- |
| base_model | The base model that we used. We use the model provided by Huggingface. (We only support LLaMA series)  |
| data_path | The path of `science_qa.hf`|
| batch_size/micro_batch_size | The micro_batch_size should be less than batch_size. |
| num_epochs/learning_rate | Training settings|
| lora_r/lora_alpha/lora_dropout | Each expert's LoRA settings|
| lora_target_modules | The modules applied to our MoLA, which can be chosen from (`q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj`) and should be separated by comma.|
| number_experts | The number of experts for each layer, which contains 32 numbers (`2,2,2,2,2,2,2,2,4,4,4,4,4,4,4,4,6,6,6,6,6,6,6,6,8,8,8,8,8,8,8,8`)|
| top_k | The top K value for each layer, which contains 32 numbers (`2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2`)|
| resume_from_checkpoint | Path of the trained MoLA model used for continuous training.|
| obalance | The usage of balance loss.|

Training on sample data:
```bash
python mola_training.py \
         --base_model "NousResearch/Llama-2-7b-hf" \
         --data_path "./sampled_data/sampled_scienceqa_train_all.hf" \
         --output_dir "./sampled_scienceqa_mola" \
         --batch_size 128 \
         --micro_batch_size 8 \
         --num_epochs 1 \
         --learning_rate 3e-4 \
         --cutoff_len 256 \
         --val_set_size 1 \
         --lora_r "8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8" \
         --lora_alpha 16 \
         --lora_dropout 0.05 \
         --lora_target_modules "q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj" \
         --number_experts "2,2,2,2,2,2,2,2,4,4,4,4,4,4,4,4,6,6,6,6,6,6,6,6,8,8,8,8,8,8,8,8" \
         --top_k "2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2" \
         --train_on_inputs \
         --group_by_length \
         --add_eos_token 
```

Training on ScienceQA data:
```bash
python mola_training.py \
         --base_model "NousResearch/Llama-2-7b-hf" \
         --data_path "./scienceqa/science_qa.hf" \
         --output_dir "./scienceqa_mola" \
         --batch_size 128 \
         --micro_batch_size 8 \
         --num_epochs 20 \
         --learning_rate 3e-4 \
         --cutoff_len 256 \
         --val_set_size 1 \
         --lora_r "8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8" \
         --lora_alpha 16 \
         --lora_dropout 0.05 \
         --lora_target_modules "q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj" \
         --number_experts "2,2,2,2,2,2,2,2,4,4,4,4,4,4,4,4,6,6,6,6,6,6,6,6,8,8,8,8,8,8,8,8" \
         --top_k "2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2" \
         --train_on_inputs \
         --group_by_length \
         --add_eos_token 
```

### Evaluation on ScienceQA (`evaluation_scienceqa.py`)
We support the evaluation of batch samples.
```bash
python evaluation_scienceqa.py \
         --test_dataset "./scienceqa/scienceq_test.json" \
         --base_model "NousResearch/Llama-2-7b-hf" \
         --mola_weights "./scienceqa_mola" \
         --batch_size 8 \
         --lora_target_modules "q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj" \
         --number_experts "2,2,2,2,2,2,2,2,4,4,4,4,4,4,4,4,6,6,6,6,6,6,6,6,8,8,8,8,8,8,8,8" \
         --top_k "2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2" \
         --save_path "./results/mola_test_sciqa.json"
```

## Acknowlegements
The code is developed based on MoLA, Huggingface, mm-cot, and alpaca-lora projects.
