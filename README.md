# Code repository for the 2-simplicial Transformer

This is the public repository for the paper "Logic and the 2-Simplicial Transformer" by James Clift, Dmitry Doryn, Daniel Murfet and James Wallbridge. The initial release contains the simplicial and relational agents, environment, training notebooks and videos of rollouts of the trained agents. This is **research code** and **some assembly may be required**. If you have problems getting the code to run, or want to request additional data not provided here, please [email Daniel](mailto:d.murfet@unimelb.edu.au).

- Notebooks for running experiments
- notes-implementation.md
- training guide
- Ray version

## Trained agent weights

In the `experiments` folder we collect some checkpoints of the eight agents described in the paper. Reconstructing the agent from these checkpoints requires some expertise with Ray RLlib.

* simplicial agent A = 30-7-19-A
* simplicial agent B = 1-8-19-A
* simplicial agent C = 23-7-19-A
* simplicial agent D = 13-8-19-C
* relational agent A = 4-8-19-A
* relational agent B = 12-6-19-A
* relational agent C = 13-8-19-A
* relational agent D = 13-6-19-C

For some of the agents the very last checkpoint is "bad", in the sense that the winrate decreased from its converged value (this is due to our use of a fixed learning rate over the entire course of training), and we are distributing the last good checkpoint, as well as a sample of earlier checkpoints. We are happy to share the entire checkpoint history, but these files approach 500Mb for some of the agents and we do not currently have a good distribution method. Nonetheless if you want the files [get in touch](mailto:d.murfet@unimelb.edu.au) and we can work something out.

## Videos

The video rollouts are provided for the best training run of the simplicial agent (simplicial agent A of the paper). The videos are organised by puzzle type, with 335C meaning the third episode sampled on puzzle type 335. Videos are not cherry-picked, and include episodes where the agent opens the bridge. There are three episodes of every puzzle type, and extras for the harder puzzles 335, 336. Figure 6 of the paper is step 8 of episode 335A, Figure 7 is step 18 of 325C, Figure 8 is step 13 of episode 335A, Figure 9 is step 29 of episode 335E.

* Solution length one: [112A](https://youtu.be/Nhvo6awWJTw), [112B](https://youtu.be/lzXRl_EyKJU), [112C](https://youtu.be/vvDDLIUztic)
* Solution length two: [213A](https://youtu.be/ZNPJJ9Iw6z8), [213B](https://youtu.be/ucWHMK0Oqoc), [213C](https://youtu.be/hZ9prVhuI7Y), [214A](https://youtu.be/A4kyF8V8178), [214B](https://youtu.be/8mOPPufnbUM), [214C](https://youtu.be/NHMNtCtfVSM), [223A](https://youtu.be/Y49_RrWTrGc), [223B](https://youtu.be/rMURm7WxCyk), [223C](https://youtu.be/k441D5WjffI), [224A](https://youtu.be/KmgTFTmJGv0), [224B](https://youtu.be/wtvwf4vNbhM), [224C](https://youtu.be/n9cH9cyF3nk)
* Solution length three: [314A](https://youtu.be/EeRM8KhyS5I), [314B](https://youtu.be/WUxuuDNECqk), [314C](https://youtu.be/YyXQTlWF50w), [315A](https://youtu.be/VvzhtOmd8w8), [315B](https://youtu.be/gsvaYCsc1Rw), [315C](https://youtu.be/2cQhaipYy30), [316A](https://youtu.be/ZfySuKCjFA8), [316B](https://youtu.be/Ew9CF5ozVq8), [316C](https://youtu.be/qofJIoIznSg), [324A](https://youtu.be/yLofHE5mHIg), [324B](https://youtu.be/dM7vg-VJOC4), [324C](https://youtu.be/soIxq-SDyBk), [325A](https://youtu.be/iCaPmkDfYmQ), [325B](https://youtu.be/EIJ5ozHk-VM), [325C](https://youtu.be/RDcg7un5u9U), [326A](https://youtu.be/PjJ3TwoZEs0), [326B](https://youtu.be/gkqLz3XlquU), [326C](https://youtu.be/-ReOqM5B_24), [334A](https://youtu.be/c_w-PFd1Fr0), [334B](https://youtu.be/fbKppQ6vNLk), [334C](https://youtu.be/hWGPjJtwtoA)
* 335: [335A](https://youtu.be/tJKjYZsenfE), [335B](https://youtu.be/pXu1OxzKJLQ), [335C](https://youtu.be/KYHFVW-tp8c), [335D](https://youtu.be/ptjdiGM1cXw), [335E](https://youtu.be/uF9sTAXahDE), [335F](https://youtu.be/RZ9kAT_bhRw), [335G](https://youtu.be/td7lwD8ujqk), [335H](https://youtu.be/kOpPEnjDIfU), [335I](https://youtu.be/LB-EmL2n78E)
* 336: [336A](https://youtu.be/m_aXi3fEqzE), [335B](https://youtu.be/lWZUDBTsRw0), [335C](https://youtu.be/HYiSM4aRBiQ), [335D](https://youtu.be/G_jnTAr_8cc)

## Caveats

- The current implementation of the simplicial agent `agent_simplicial.py` assumes one head of 2-simplicial attention.
