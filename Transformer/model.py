import numpy as np

def layer_norm(x, eps=1e-6):
    mean = np.mean(x, axis=-1, keepdims=True)
    std = np.std(x, axis=-1, keepdims=True)
    gamma = np.ones_like(mean)  # Scale parameter (initialized to 1)
    beta = np.zeros_like(mean)  # Shift parameter (initialized to 0)
    return gamma * (x - mean) / (std + eps) + beta

def feed_forward_network(x, d_model, d_ff):
    W1 = np.random.rand(d_model, d_ff)
    W2 = np.random.rand(d_ff, d_model)
    b1 = np.random.rand(d_ff)
    b2 = np.random.rand(d_model)
    return np.dot(np.maximum(0, np.dot(x, W1) + b1), W2) + b2

def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    scores = np.matmul(Q, K.T) / np.sqrt(d_k)
    
    if mask is not None:
        scores = np.where(mask == 0, -1e9, scores)
        
    attention_weights = softmax(scores)
    output = np.matmul(attention_weights, V)
    return output, attention_weights

def multi_head_attention(x, num_heads, d_model, mask=None):
    d_k = d_v = d_model // num_heads
    
    W_Q = [np.random.rand(d_model, d_k) for _ in range(num_heads)]
    W_K = [np.random.rand(d_model, d_k) for _ in range(num_heads)]
    W_V = [np.random.rand(d_model, d_v) for _ in range(num_heads)]
    W_O = np.random.rand(num_heads * d_v, d_model)
    
    heads = []
    for i in range(num_heads):
        Q = np.dot(x, W_Q[i])
        K = np.dot(x, W_K[i])
        V = np.dot(x, W_V[i])
        head, _ = scaled_dot_product_attention(Q, K, V, mask)
        heads.append(head)
    
    concatenated_heads = np.concatenate(heads, axis=-1)
    output = np.dot(concatenated_heads, W_O)
    
    return output

def masked_multi_head_attention(x, num_heads, d_model):
    mask = np.triu(np.ones((x.shape[1], x.shape[1])), k=1).astype(np.uint8)
    return multi_head_attention(x, num_heads, d_model, mask)

def decoder_layer(x, encoder_output, d_model, num_heads, d_ff):
    masked_attn_output = masked_multi_head_attention(x, num_heads, d_model)
    out1 = layer_norm(x + masked_attn_output)
    
    attn_output = multi_head_attention(out1, num_heads, d_model, encoder_output)
    out2 = layer_norm(out1 + attn_output)
    
    ffn_output = feed_forward_network(out2, d_model, d_ff)
    out3 = layer_norm(out2 + ffn_output)
    
    return out3

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

def final_linear_projection(decoder_output, vocab_size):
    d_model = decoder_output.shape[-1]
    W_out = np.random.rand(d_model, vocab_size)
    b_out = np.random.rand(vocab_size)
    return np.dot(decoder_output, W_out) + b_out

def output_layer(decoder_output, vocab_size):
    # Linear projection to vocabulary size
    logits = final_linear_projection(decoder_output, vocab_size)
    
    # Apply softmax to get probabilities
    probabilities = softmax(logits)
    
    return probabilities

# Example usage
np.random.seed(0)  # For reproducibility

sequence_length = 10
d_model = 512
num_heads = 8
d_ff = 2048
vocab_size = 10000  # Example vocabulary size

decoder_input_embedding = np.random.rand(sequence_length, d_model)
encoder_output = np.random.rand(sequence_length, d_model)

decoder_output = decoder_layer(decoder_input_embedding, encoder_output, d_model, num_heads, d_ff)

# Apply the output layer
output_probabilities = output_layer(decoder_output, vocab_size)

print("Output probabilities shape:\n", output_probabilities.shape)
print("Output probabilities:\n", output_probabilities)
