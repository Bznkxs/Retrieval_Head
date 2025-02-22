import json
import numpy as np


# te_ln = json.load(open("/u/yufengd4/te_layer_norm_out_fp16.txt"))
# faiss_ln = json.load(open("/u/yufengd4/faiss_ln_output_fp16.txt"))
#
# te_ln = np.array(te_ln)
# faiss_ln = np.array(faiss_ln)[0]

def compare(te_ln, faiss_ln):
    print("Comparing layer norms...")
    # print(len(te_ln), len(faiss_ln))
    # print(len(te_ln[0]), len(faiss_ln[0]))
    print(te_ln.shape, faiss_ln.shape)
    print(np.sqrt((np.abs(te_ln - faiss_ln) ** 2).sum()))


# te_qkv = json.load(open("/u/yufengd4/te_qkv_output_fp16.txt"))
# faiss_qkv = json.load(open("/u/yufengd4/faiss_qkv_output_fp16.txt"))

# te_qkv = np.array(te_qkv)
# faiss_qkv = np.array(faiss_qkv)

def compare_qkv(te_w, faiss_w):
    print("Comparing weights...")
    print(te_w.shape, faiss_w.shape)
    print(np.sqrt((np.abs(te_w - faiss_w) ** 2).sum()))

# faiss_ln_input = json.load(open("/u/yufengd4/te_layer_norm_input.txt"))
# te_ln_input = json.load(open("/u/yufengd4/te_inputmat.txt"))
# te_ln_input = np.array(te_ln_input)
# faiss_ln_input = np.array(faiss_ln_input)

def compare_input(te_ln_input, faiss_ln_input):
    print("Comparing layer norm inputs...")

    print(te_ln_input.shape, faiss_ln_input.shape)
    print(np.sqrt((np.abs(te_ln_input - faiss_ln_input) ** 2).sum()))


# f_ln_b = json.load(open("/u/yufengd4/f_llama_input_layer_norm.txt"))
# te_ln_b = json.load(open("/u/yufengd4/te_layer_norm_out1.txt"))
# f_ln_b = np.array(f_ln_b)
# te_ln_b = np.array(te_ln_b)

# faiss_layer_out = json.load(open("/u/yufengd4/faiss_layer_norm_hidden_states_fp16.txt"))  # this is output of layer norm
# te_layer_in = json.load(open("/u/yufengd4/megatron_layer_norm_hidden_states_fp16.txt"))  # this is input of layer norm
# faiss_layer_out = np.array(faiss_layer_out)
# te_layer_in = np.array(te_layer_in)
# faiss_layer_in = json.load(open("/u/yufengd4/faiss_input_hidden_states_fp16.txt"))  # this is input of layer norm
# te_layer_out = json.load(open("/u/yufengd4/te_layer_norm_out_fp16.txt"))  # this is output of layer norm
# faiss_layer_in = np.array(faiss_layer_in)
# te_layer_out = np.array(te_layer_out)
#
# te_post_rope_q = json.load(open("/u/yufengd4/megatron_post_rope_query_fp16.txt"))
# faiss_post_rope_q = json.load(open("/u/yufengd4/faiss_post_rope_query.txt"))
# te_post_rope_q = np.array(te_post_rope_q)
# faiss_post_rope_q = np.array(faiss_post_rope_q)
# te_post_rope_q = te_post_rope_q.transpose(1, 2, 0, 3)
# compare_input(te_post_rope_q, faiss_post_rope_q[:, :8, :, :])
# te=json.load(open("/u/yufengd4/te_attn_out_fp16.txt"))
# faiss=json.load(open("/u/yufengd4/faiss_attn_out_fp16.txt"))

for i in range(32):
    te_output_hidden_states = json.load(open(f"/u/yufengd4/te_output_hidden_states_{i}.txt"))
    faiss_output_hidden_states = json.load(open(f"/u/yufengd4/faiss_output_hidden_states_{i}.txt"))
    te_output_hidden_states = np.array(te_output_hidden_states).reshape(-1, 4096)
    faiss_output_hidden_states = np.array(faiss_output_hidden_states).reshape(-1, 4096)
    compare_input(te_output_hidden_states, faiss_output_hidden_states)
