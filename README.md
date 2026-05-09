# recollection.local

> Local Recollection, RAG 


# High Level 
> Agentic reasoning under strict VRAM constraints via confidence-gated computation.
Through __co-design of inference algorithm + systems constraints__

## Mainly targeting 3 main pain points 
1. LIM : Lost in the Middle 
2. RAG noise / fragmentation 
3. KV cache explosion for agentic methods 


## Main aspects 
1. Confidence Gated Discrimination 
    - Cuts of unnecessary reasoning paths 
    - Reasoning is a conditional compute but not alwasys on
    - Simple implementation 

> `Don't think harder unless you're unsure`

## Co-design Aspects 
> of systems + algorithm 
- INT4 + PolarQuant + KV compression 
- bounded MCTS 
- retrieval gating 

## Framing 
- Core idea is confidence-gated reasoning, supporting systems are quantization and retrieval choices 


# Things that I don't like 
1. Is _confidence_ reliable enough to gate reasoning?
    - LMs are often overconfident when wrong (esp. multi-hop reasoning)
    - Solutions in my head 
        1. Can show correlation ~ conficence vs correctness 
        2. Fallback 💀
            - periodic forced retrievals 
            - uncertainty smoothing 



- - - - - - - - - - 

# Benchmarks 
1. NeedleChain 
2. LongBench v2 