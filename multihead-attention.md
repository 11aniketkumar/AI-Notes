[Transformer](./index.md) > MultiHead Attention

# MultiHead Attention
## The Need for Multiple Perspectives
As we discovered in the previous chapter, standard single-head self-attention forces a model to view a sentence through a single lens, struggling with ambiguities where multiple contextual relationships coexist simultaneously. 

To overcome this limitation, we use **Multi-Head Attention**. Instead of performing a single attention function with $d_{model}$-dimensional queries, keys, and values, multi-head attention projects the queries, keys, and values $h$ times with different, learned linear projections to lower dimensions. 

## Mechanics of Multi-Head Attention
- We run several attention "heads" in parallel, allowing the model to jointly attend to information from different representation subspaces at different positions.
- After computing attention independently across all heads, we concatenate their resulting outputs and pass them through a final linear projection. 
- At the end of this pipeline, the output embedding size remains identical to the input embedding size.
![](images/multihead%20self%20attention.png)

## Limitations of Attention and Next Steps
While multi-head attention successfully captures rich semantic and syntactic relationships across various perspectives, it introduces a new structural blind spot: **parallel processing lacks sequence awareness**. 

Because every token interacts with every other token simultaneously, the model treats the input like an unordered bag of vectors, losing all track of word order and positional layout. In the next chapter, we will explore how positional information is re-introduced into the architecture.