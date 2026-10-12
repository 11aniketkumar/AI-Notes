[Transformer](./index.md) > Positional Encoding

# Positional Encoding
- Positions of words can be encoded into embeddings by adding a few bits at the last, but it creates a few problems:
- Increased dimension increases compute requirement.
- Unbounded - no upper limit.
- Discrete - neural networks like continuous functions.

### The Sinusoidal Addition Method
- One of the ways that we can use is to create a positional vector using sine and cosine waves, and then add them to the embedding before passing them to the attention block.
- Sine and cos are continuous and bounded.
- The positional encoding must be of the same size as that of the embedding so that we can add them.
- We need to make sure that each position is unique, no two positions can have the same positional encoding, therefore we take multiple sine and cosine pairs instead of just one.

$$\sin(pos), \cos(pos), \sin\left(\frac{pos}{2}\right), \cos\left(\frac{pos}{2}\right), \dots$$

#### The Transformer Formulation

Instead of simple divisors, the original paper defines the positional encoding equations for each dimension index $i$:

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

### Visualizing the Positional Pattern
- If you plot the equations on a graph, you get something like this:
![](images/positional%20encoding.png)
- This is quite interesting, as if you notice closely it creates a binary type of pattern where first initial bits are getting affected while higher bits remain unaffected for smaller values.
- This is one of the reasons why adding positional encoding to embedding doesn't affect meaning as 512 dimensions is a big space, the chances of colliding with another valid embedding due to some small change is almost zero.
> **Analogy:** Think of changing a speaker's voice from male to female, or adjusting pitch and speaking speed slightly. The acoustic properties shift, but the underlying meaning of the words remains completely intact. Similarly, positional encoding adds a minor structural fingerprint without altering semantic representation.