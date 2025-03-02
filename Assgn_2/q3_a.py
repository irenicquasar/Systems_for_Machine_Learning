import math
import random

################################################################################
#                           16-BIT FLOAT EMULATION
################################################################################

def float_to_half_precision(value):
    """
    Emulate half precision by rounding to a few decimal places.
    In a real system, you would convert to IEEE 754 binary16.
    For demonstration, we just round to 3 decimals.
    """
    return round(value, 3)

################################################################################
#                         BASIC LINEAR ALGEBRA HELPERS
################################################################################

def create_random_vector(size):
    """Create a random vector of given size, each entry in half precision."""
    return [float_to_half_precision(random.random()) for _ in range(size)]

def create_random_matrix(rows, cols):
    """Create a random matrix (rows x cols) in half precision."""
    return [create_random_vector(cols) for _ in range(rows)]

def matmul(A, B):
    """
    Matrix multiplication in pure Python with half-precision rounding.
    A: shape (r1, c1)
    B: shape (r2, c2) with r2 == c1
    Returns: shape (r1, c2)
    """
    r1 = len(A)
    c1 = len(A[0])
    r2 = len(B)
    c2 = len(B[0])
    assert c1 == r2, "Inner dimension mismatch in matmul."
    
    result = []
    for i in range(r1):
        row_result = []
        for j in range(c2):
            s = 0.0
            for k in range(c1):
                s += A[i][k] * B[k][j]
            # Round to half precision
            row_result.append(float_to_half_precision(s))
        result.append(row_result)
    return result

def add_matrices(A, B):
    """
    Element-wise matrix addition, storing in half precision.
    A, B: same shape
    """
    r = len(A)
    c = len(A[0])
    result = []
    for i in range(r):
        row = []
        for j in range(c):
            val = A[i][j] + B[i][j]
            row.append(float_to_half_precision(val))
        result.append(row)
    return result

def transpose(matrix):
    """Transpose of a 2D list."""
    rows = len(matrix)
    cols = len(matrix[0])
    result = []
    for c in range(cols):
        new_row = []
        for r in range(rows):
            new_row.append(matrix[r][c])
        result.append(new_row)
    return result

################################################################################
#                           NORMALIZATION & SOFTMAX
################################################################################

def layer_norm(X):
    """
    Simple Layer Normalization across the last dimension (features).
    X shape: (seq_len, d_model)
    """
    eps = 1e-5
    normalized = []
    for row in X:
        mean_val = sum(row) / len(row)
        var_val = sum((x - mean_val)**2 for x in row) / len(row)
        stdev = math.sqrt(var_val + eps)
        norm_row = [float_to_half_precision((x - mean_val)/stdev) for x in row]
        normalized.append(norm_row)
    return normalized

def softmax(vec):
    """
    Standard softmax over a 1D vector, with half-precision output.
    """
    # Subtract max for numerical stability
    max_val = max(vec)
    exps = [math.exp(v - max_val) for v in vec]
    s = sum(exps)
    return [float_to_half_precision(e / s) for e in exps]

################################################################################
#                       MULTI-HEAD ATTENTION SUBROUTINES
################################################################################

def split_heads(X, num_heads):
    """
    Split X of shape (seq_len, d_model) into (num_heads, seq_len, head_dim).
    head_dim = d_model // num_heads.
    """
    seq_len = len(X)
    d_model = len(X[0])
    head_dim = d_model // num_heads
    heads = []
    for h in range(num_heads):
        head_matrix = []
        for i in range(seq_len):
            # slice from h*head_dim to (h+1)*head_dim
            head_matrix.append(X[i][h*head_dim : (h+1)*head_dim])
        heads.append(head_matrix)
    return heads

def combine_heads(heads):
    """
    Inverse of split_heads. 
    heads shape: (num_heads, seq_len, head_dim)
    Return shape: (seq_len, num_heads * head_dim).
    """
    num_heads = len(heads)
    seq_len = len(heads[0])
    head_dim = len(heads[0][0])
    
    combined = []
    for i in range(seq_len):
        row = []
        for h in range(num_heads):
            row.extend(heads[h][i])
        combined.append(row)
    return combined

def attention(Q, K, V, mask=None):
    """
    Scaled dot-product attention for a single head.
    Q, K, V: shape (seq_len, head_dim)
    mask: shape (seq_len, seq_len), with 1=keep, 0=mask, or None
    Returns: shape (seq_len, head_dim).
    """
    seq_len = len(Q)
    head_dim = len(Q[0])
    
    # Compute QK^T / sqrt(head_dim)
    K_t = transpose(K)  # shape (head_dim, seq_len)
    
    scores = []
    for i in range(seq_len):
        row_scores = []
        for j in range(seq_len):
            dot = 0.0
            for k in range(head_dim):
                dot += Q[i][k] * K[j][k]
            # Scale
            dot /= math.sqrt(head_dim)
            
            # If mask is given and mask[i][j] == 0, set large negative
            if mask is not None and mask[i][j] == 0:
                dot = -1e9
            
            row_scores.append(dot)
        scores.append(row_scores)
    
    # Softmax row by row
    attn_weights = [softmax(row) for row in scores]
    
    # Multiply by V
    out = []
    for i in range(seq_len):
        out_row = [0.0]*head_dim
        for j in range(seq_len):
            w = attn_weights[i][j]
            for k in range(head_dim):
                out_row[k] += w * V[j][k]
        # Round to half precision
        out_row = [float_to_half_precision(x) for x in out_row]
        out.append(out_row)
    
    return out

def multi_head_attention(X, Wq, Wk, Wv, Wo, num_heads=8, mask=None):
    """
    Multi-head attention wrapper.
    X: shape (seq_len, d_model)
    Wq, Wk, Wv: (d_model, d_model)
    Wo: (d_model, d_model)
    mask: optional (seq_len x seq_len)
    """
    # 1) Compute Q, K, V
    Q = matmul(X, Wq)  # (seq_len, d_model)
    K = matmul(X, Wk)
    V = matmul(X, Wv)
    
    # 2) Split into heads
    Q_heads = split_heads(Q, num_heads)
    K_heads = split_heads(K, num_heads)
    V_heads = split_heads(V, num_heads)
    
    # 3) Scaled dot-product attention per head
    head_outputs = []
    for h in range(num_heads):
        head_out = attention(Q_heads[h], K_heads[h], V_heads[h], mask)
        head_outputs.append(head_out)
    
    # 4) Combine heads
    combined = combine_heads(head_outputs)  # (seq_len, d_model)
    
    # 5) Final linear projection
    out = matmul(combined, Wo)  # (seq_len, d_model)
    return out

################################################################################
#                         ENCODER & DECODER BLOCKS
################################################################################

def encoder_block(X, Wq, Wk, Wv, Wo):
    """
    Single encoder block with:
      Multi-head attention -> Add & Norm
    (Feed-forward is skipped in this modified Transformer.)
    """
    # Multi-head self-attention
    attn_out = multi_head_attention(X, Wq, Wk, Wv, Wo, num_heads=8, mask=None)
    
    # Add & Norm
    out1 = add_matrices(X, attn_out)
    norm_out = layer_norm(out1)
    return norm_out

def decoder_block(X, enc_out,
                  Wq_self, Wk_self, Wv_self, Wo_self,
                  Wq_enc,  Wk_enc,  Wv_enc,  Wo_enc):
    """
    Single decoder block with:
      1) Masked Multi-head self-attention -> Add & Norm
      2) Cross-attention with encoder -> Add & Norm
    (Feed-forward is skipped in this modified Transformer.)
    """
    seq_len = len(X)
    
    # 1) Masked multi-head attention (causal mask)
    causal_mask = []
    for i in range(seq_len):
        row = []
        for j in range(seq_len):
            # 1 if j <= i, else 0
            row.append(1 if j <= i else 0)
        causal_mask.append(row)
    
    attn_out_self = multi_head_attention(X, Wq_self, Wk_self, Wv_self,
                                         Wo_self, num_heads=8, mask=causal_mask)
    out1 = add_matrices(X, attn_out_self)
    norm_out1 = layer_norm(out1)
    
    # 2) Cross-attention: Q = norm_out1, K=enc_out, V=enc_out
    
    # Q = norm_out1 * Wq_enc
    Q = matmul(norm_out1, Wq_enc)
    # K = enc_out * Wk_enc
    K = matmul(enc_out, Wk_enc)
    # V = enc_out * Wv_enc
    V = matmul(enc_out, Wv_enc)
    
    # We'll define a small helper for cross-attention
    def cross_attention(Q, K, V, Wo, num_heads=8):
        """
        Multi-head cross-attention: Q from decoder side, K,V from encoder side.
        Q shape: (dec_len, d_model)
        K,V shape: (enc_len, d_model)
        """
        Q_heads = split_heads(Q, num_heads)  # (num_heads, dec_len, head_dim)
        K_heads = split_heads(K, num_heads)  # (num_heads, enc_len, head_dim)
        V_heads = split_heads(V, num_heads)  # (num_heads, enc_len, head_dim)
        
        head_outputs = []
        for h in range(num_heads):
            head_outputs.append(cross_attention_single_head(Q_heads[h],
                                                            K_heads[h],
                                                            V_heads[h]))
        
        combined = combine_heads(head_outputs)  # (dec_len, d_model)
        return matmul(combined, Wo)
    
    def cross_attention_single_head(Qh, Kh, Vh):
        """
        Single-head cross attention with Qh shape (dec_len, head_dim),
        Kh, Vh shape (enc_len, head_dim).
        """
        dec_len = len(Qh)
        enc_len = len(Kh)
        head_dim = len(Qh[0])
        
        # QK^T / sqrt(head_dim)
        Kh_t = transpose(Kh)
        
        scores = []
        for i in range(dec_len):
            row_scores = []
            for j in range(enc_len):
                dot = 0.0
                for c in range(head_dim):
                    dot += Qh[i][c] * Kh[j][c]
                dot /= math.sqrt(head_dim)
                row_scores.append(dot)
            scores.append(row_scores)
        
        # softmax row-wise
        attn_weights = [softmax(row) for row in scores]
        
        # multiply by V
        out = []
        for i in range(dec_len):
            out_row = [0.0]*head_dim
            for j in range(enc_len):
                w = attn_weights[i][j]
                for c in range(head_dim):
                    out_row[c] += w * Vh[j][c]
            out_row = [float_to_half_precision(x) for x in out_row]
            out.append(out_row)
        return out
    
    attn_out_enc = cross_attention(Q, K, V, Wo_enc, num_heads=8)
    out2 = add_matrices(norm_out1, attn_out_enc)
    norm_out2 = layer_norm(out2)
    
    return norm_out2

################################################################################
#                           RANDOM EMBEDDING FOR WORDS
################################################################################

def sentence_to_random_embeddings(sentence, d_model=256):
    """
    Take a raw sentence (string) with words separated by spaces.
    Return a list of shape (seq_len, d_model) with random embeddings
    for each word. We do NOT use a real tokenizer or dictionary here:
    each word is simply mapped to a new random vector for demonstration.
    """
    words = sentence.strip().split()
    X = []
    for w in words:
        # Create a random 256-dim vector in half precision
        row = [float_to_half_precision(random.random()) for _ in range(d_model)]
        X.append(row)
    return X

def positional_encoding(X):
    """
    Add standard sinusoidal positional encoding to X in-place.
    X shape: (seq_len, d_model).
    """
    seq_len = len(X)
    if seq_len == 0:
        return X
    d_model = len(X[0])
    
    for pos in range(seq_len):
        for i in range(d_model):
            if i % 2 == 0:
                val = math.sin(pos / (10000 ** (2 * i / d_model)))
            else:
                val = math.cos(pos / (10000 ** (2 * (i - 1) / d_model)))
            X[pos][i] = float_to_half_precision(X[pos][i] + val)
    return X

################################################################################
#                         FULL ENCODER-DECODER FORWARD
################################################################################

def transformer_forward_sentences(input_sentence, output_sentence):
    """
    Forward pass of the (modified) Transformer with:
      - 1 Encoder block
      - 1 Decoder block
      - Final linear + softmax
    Using random word embeddings (256-dim) for each word, and random weights,
    all in half precision.
    
    input_sentence, output_sentence: raw strings (e.g. "Hello how are you")
    Returns: Probability distributions for each time step in the decoder output.
    """
    d_model = 256
    num_heads = 8
    
    #--------------------------------------------------------------------------
    # 1) Convert input sentence to random embeddings + positional encoding
    #--------------------------------------------------------------------------
    X_in = sentence_to_random_embeddings(input_sentence, d_model=d_model)
    X_in = positional_encoding(X_in)  # add sinusoidal PE
    
    #--------------------------------------------------------------------------
    # 2) Encoder block
    #--------------------------------------------------------------------------
    # Create random weights for the encoder MHA
    Wq_enc = create_random_matrix(d_model, d_model)
    Wk_enc = create_random_matrix(d_model, d_model)
    Wv_enc = create_random_matrix(d_model, d_model)
    Wo_enc = create_random_matrix(d_model, d_model)
    
    enc_out = encoder_block(X_in, Wq_enc, Wk_enc, Wv_enc, Wo_enc)
    
    #--------------------------------------------------------------------------
    # 3) Convert output sentence to random embeddings + positional encoding
    #--------------------------------------------------------------------------
    X_out = sentence_to_random_embeddings(output_sentence, d_model=d_model)
    X_out = positional_encoding(X_out)
    
    #--------------------------------------------------------------------------
    # 4) Decoder block
    #--------------------------------------------------------------------------
    # Weights for the decoder self-attn
    Wq_self = create_random_matrix(d_model, d_model)
    Wk_self = create_random_matrix(d_model, d_model)
    Wv_self = create_random_matrix(d_model, d_model)
    Wo_self = create_random_matrix(d_model, d_model)
    
    # Weights for the cross-attn (encoder-decoder)
    Wq_enc_dec = create_random_matrix(d_model, d_model)
    Wk_enc_dec = create_random_matrix(d_model, d_model)
    Wv_enc_dec = create_random_matrix(d_model, d_model)
    Wo_enc_dec = create_random_matrix(d_model, d_model)
    
    dec_out = decoder_block(X_out, enc_out,
                            Wq_self, Wk_self, Wv_self, Wo_self,
                            Wq_enc_dec, Wk_enc_dec, Wv_enc_dec, Wo_enc_dec)
    
    #--------------------------------------------------------------------------
    # 5) Final Linear + Softmax
    #--------------------------------------------------------------------------
    # Typically you'd project to the full vocab. We'll do a small dimension = 10
    W_final = create_random_matrix(d_model, d_model)
    logits = matmul(dec_out, W_final)  # shape (seq_len_out, d_model)
    
    small_vocab_size = 10
    W_vocab = create_random_matrix(d_model, small_vocab_size)
    
    final_logits = matmul(logits, W_vocab)  # shape (seq_len_out, 10)
    
    # Softmax for each time step
    prob_out = []
    for row in final_logits:
        prob_out.append(softmax(row))
    
    return prob_out


################################################################################
#                               MAIN FUNCTION
################################################################################
def main():
    random.seed(0)
    
    # Example: pass raw sentences (no tokenizer)
    input_sentence = "hello how are you"
    output_sentence = "i am fine sure"
    
    # Forward pass
    probabilities = transformer_forward_sentences(input_sentence, output_sentence)
    
    print("Final output probabilities per decoder time-step (for a 10-word 'vocab'):")
    for t, row in enumerate(probabilities):
        print(f"Time step {t}: {row}")

if __name__ == "__main__":
    main()
