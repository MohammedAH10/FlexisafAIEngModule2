# UNDRERSTANDING LLMS Flexisaf AI Engineering Week2

Tokens and attention are the two foundational ideas inside every transformer. A token is not a word it is a chunk produced by a subword tokenizer (BPE or similar),
so "unbelievable" might split into ["un", "believ", "able"]. Each token is mapped to a high-dimensional embedding vector, carrying an initial semantic position in a 
learned space. Attention is then the mechanism by which tokens update those vectors by looking at each other. For each token acting as a query, the model computes a 
dot product against every other token acting as a key, scales and softmaxes those scores into a probability distribution (the attention weights), and uses them to 
form a weighted sum of value vectors. The result is a new, context-aware representation of that token — one that has "borrowed" meaning from other relevant tokens.
Transformer blocks stack this operation repeatedly. Each block is: multi-head self-attention (running several attention patterns in parallel, each learning a different 
relational structure), followed by a position-wise feed-forward network (a two-layer MLP that applies a nonlinear transformation independently to each token's vector), 
with residual connections and layer normalization wrapping both. After N such blocks, the final token representations are projected to vocabulary logits and a softmax 
gives the next-token probability distribution. The depth of the stack is what allows the model to compose increasingly abstract representations — early layers tend to 
capture syntactic structure, later layers semantic and world-knowledge relationships.
