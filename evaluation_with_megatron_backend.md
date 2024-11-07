# Evaluation of Needle-in-a-Haystack with Megatron Backend

## Environment Setup

### Install Megatron Environment

Clone our Megatron-LM fork. Switch to the `tokenize_diverge` branch. 
Follow the instructions in the Megatron-LM repository. It can be complicated and time-consuming.

**Install flask for serving the model.** `pip install flask-restful`

### Install NIHS Environment

Clone this repository. Install `transformers` and `rouge_score`.

### Convert a model to Megatron format

Follow the instructions in the Megatron-LM repository to convert a model to the Megatron format. Or for a quick conversion:

```bash
python /home/user1/Megatron-LM/tools/checkpoint/convert.py \
    --model-type GPT \
    --loader llama_mistral \
    --saver mcore \
    --megatron-path /home/user1/Megatron-LM \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 1 \
    --load-dir /work/user1/huggingface_models/Llama-2-7B-hf \
    --save-dir /work/user1/megatron_models/Llama-2-7B-hf-Megatron
```

### Launch the Inference Server

In a node with both CUDA environment and intra-net access, modify the launch script (ignore the llama3 name)

```
/home/user1/Megatron-LM/examples/inference/run_text_generation_llama3.sh
```

and launch the server with two arguments, the Megatron model directory and the directory where your huggingface tokenizer is stored. For example, if the model is stored in `/work/user1/megatron_models/Llama-2-7B-hf-Megatron` and the tokenizer is stored:

```bash
bash /home/user1/Megatron-LM/examples/inference/run_text_generation_llama3.sh \
    /work/user1/megatron_models/Llama-2-7B-hf-Megatron \
    /work/user1/huggingface_models/Llama-2-7B-hf
```

After the server is launched, it tells you its url. The server will be running on port 5000 by default.

### Evaluate the Model

Change your directory to this repository.
Use `needle_in_a_haystack_megatron.py` to evaluate the model. It is easy to figure out how to use it. Some args:

- `--window-size`: the sliding window size. Default is None.
- `--s`, `--e`: the start and end context sizes.
- `--num_intervals`: the number of intervals to evaluate. The total test range from s_len and e_len will be divided into this number of intervals. Default is 40.
- `--model_provider`: optional, exists due to legacy factor. you should only use the DEFAULT value `Megatron`.
- `--model_name`: more accurately stands for "experiment name". Name your experiment.
- `--url`: the url of the server. Remove "http://" from your input.

An example command is shown in the script `megatron_test.sh`. You can run it with:
```bash
bash megatron_test.sh
```