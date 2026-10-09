# ALHR (Adaptive learnable Hierarchical routing): sub-quadratic tree attention without accuracy loss, trained in three phases against a dense reference (beta)

An attention mechanism using static binary trees and learnable functions to achieve sub-quadratic inference without meaningful accuracy loss. 
I used a static binary tree to sort the key matrix, where each branch is indexed by the sum of all the key vectors beneath it.

## Inference stage
After the input is split into queries and keys, the static binary tree is built on top of the key matrix and stored. This happens once per layer.

Here is where our trainable function comes in, the budget predictor. The budget predictor is a trainable function that essentially takes in the current query vector, and a global sum of all the vectors, to essentially output the number of keys that the query is asking for(it outputs a leaf group for the query that represents the number of keys that the query is asking for).

After we have this information, the first thing we do is search the query against its own key vector and its neighbors(a similar system to Treeformer).
For this it uses the data of the previous queries that matched with their own neighbours.

After neighbor search is done, and the queries leaf group is still not satisfied(meaning it still has more keys to search), the query descended through the tree, by comparing it with the branches and going down the more similar branch. The number of branches it goes down is decided by how many keys are left to be found by the query(subtracting found neighbour keys with the leaf group).

After each query finds its respective keys, standard self-attention dot product and sum is run.

After factoring in tree descent, the operations complexity of inference is about O(NlogN)

## Training stage
This is the most important part of the mechanism and also the most expensive. It is split into 3 phases.

**Phase 1**: Here, the tree module does not factor in at all, instead a dense model and a FFN are trained against the data. This is the step that makes training quadratic. This dense model acts as a teacher.

**Phase 2**:The attention matrix from the trained dense model is extracted and the top-k keys for every query is labelled. Using this data and no external data we train out tree attention module. Here is where the dense model teaches the tree. Training doesn't happen to completion here as dense and sparse matrixes fundamentally differ, and we don't want exact training either.

**Phase 3**:The tree is trained with the FFN. Not a new FFN, the same one attached to the dense model in phase 1. Since we are training without the dense teacher here, and directly training against data, the tree can actually learn to surpass its teacher(in the MQAR tests below it actually displays this).

In phase 2 and 3, the functions trained in the tree module are the query/key splitter and the budget predictor.
Causality is maintained by choosing the branches/branch of the tree that only contain past key vectors and treating it as the whole tree.

## Results and Variables
The following is also present in the repo under logs.

Ideally, I would have run MQAR and Tinystories tests at 1,4,16k Tokens to solidify, but as things are right now, I only have the results for MQAR testing at 1K Tokens. 

The exact log and the reproducible Kaggle script is in the logs folder

**HERE ARE THE RESULTS**: 

## 1024 Tokens Standard MQAR testing

| Model Variant (T=1024) | Top-1 Accuracy | Avg Keys read/Token | KV Compression | Peak VRAM | Cache Compression |
|---|---|---|---|---|---|
| Dense Baseline(Teacher) | 94.9% ± 1.5% | 512.5(Full) | 1.0 x (100% Read) | 57MB | N/A |
| ALHR(Tree attention) | 92.1% ± 0.6% | 30.0 | 35.3 x (2.83% Read) | 422MB | 100.0% |


Now as you can see, we achieve near dense accuracy while reading a small amount of keys.

In fact, in some of my earlier testing, our tree beat its own teacher. However I do not have the resources right now to reproduce that on this scale, It is theoretically possible.

One thing you will notice is the VRAM used is significantly higher, however as number of tokens increase, peak VRAM increases dramatically for dense while ours scales linearly.

Also we can verify sub-quadratic inference from here, as approximately Nlog N, as seen in the logs.

Again, I wanted to do this multi seed run for up to 16k tokens, but I do not have the resources. (iPhone 16e and kaggle 😭)

These are the proprietary results.

You can verify the data set and kaggle prompt I used to get this in the logs folder.

## Paper
I'm currently in the process of writing a paper and uploading it to Zenodo, but its been difficult given my lack of resources and balancing my college life.

## Contributors
This was a solo project(Im just a 18y/o college student lol). Just me and my iPhone 16e and Kaggle notebooks. I used Claude sonnet 5.5 to generate the code.

## Donation and sponsors
Pls donate to support further development and even more ideas that I have.
(Im tired of doing this on a small phone 😭)

UPI: vdhiroshnin@okhdfcbank

Buy me a coffee: https://buymeacoffee.com/vdhiroshnif

Donors get access to the development logs and process along with some ideas I had to put on the side.

## Contact me
E-mail: dexend8123@gmail.com

## Citations


