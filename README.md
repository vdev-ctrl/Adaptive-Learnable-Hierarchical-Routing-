# ALHR(Adaptive learnable Hierarchical routing): sub-quadratic tree attention without accuracy loss, trained in three phases against a dense reference

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
[To be pasted here]

## Paper
I'm currently in the process of writing a paper and uploading it to Zenodo, but its been difficult given my lack of resources and balancing my college life.

## Contributors
This was a solo project(Im just a 18y/o college student lol). Just me and my iPhone 16e and Kaggle notebooks. I used Claude sonnet 5.5 to generate the code.

## Donation and sponsors
Pls donate to support further development and even more ideas that I have.
Patreon will include the development progress logs and some of the ideas that I had put on hold.

## Contact me
E-mail: dexend8123@gmail.com

## Citations


