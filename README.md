# Code repository for the 2-simplicial Transformer

This is the public repository for the paper "Logic and the 2-Simplicial Transformer" by James Clift, Dmitry Doryn, Daniel Murfet and James Wallbridge. The initial release contains the simplicial and relational agents, environment, training notebooks and videos of rollouts of the trained agents. This is **research code** and **some assembly may be required**. If you have problems getting the code to run, or want to request additional data not provided here, please [email Daniel](mailto:d.murfet@unimelb.edu.au).

Main files:

- Relational agent: `agent/agent_relational.py`
- Simplicial agent: `agent/agent_simplicial.py`
- Environment: `env/bridge_boxworld.py`
- Training notebooks: `notebooks/`
- Videos (see below)
- Trained agent weights (see below)

There is a brief training guide in `notebooks/README.md` and brief installation instructions below. In `notes-implementation.md` we collect various notes about training agents with IMPALA in Ray that might be useful (but as the Ray codebase is evolving quickly, many of the class names in these notes may now be incorrect).

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

## Installation

The following instructions assume you know how to set up TensorFlow, and cover the other aspects of setting up a blank GCP or AWS instance to a point where they can run our training notebooks. Our training was done under Ray version `0.7.0.dev2` and we do not make any assurances that the code will even run on later versions of Ray. As detailed in the paper, our head nodes (the ones on which we run the training notebooks) have either a P100 or K80 GPU, and the worker nodes have no GPU.

```
sudo apt-get update
sudo apt install python-pip
sudo apt install python3-dev python3-pip
sudo apt install cmake
sudo apt-get install zlib1g-dev
sudo apt install git
pip3 install --user tensorflow-gpu
pip3 install -U dask
pip3 install --user ray[rllib]
pip3 install --user ray[debug]
pip3 install jupyter
pip3 install -U matplotlib
pip3 install psutil
pip3 install --upgrade gym
sudo apt-get install ffmpeg
sudo apt-get install pssh
sudo apt-get install keychain
pip3 install -U https://s3-us-west-2.amazonaws.com/ray-wheels/latest/ray-0.7.0.dev2-cp36-cp36m-manylinux1_x86_64.whl
```

On the CPU-only machines use `pip3 install --user tensorflow`

More installation:

```
git clone https://github.com/ray-project/ray
git clone https://github.com/dmurfet/simplicialtransformer.git
cd ~/simplicialtransformer/; git config credential.helper store; git fetch
git clone https://github.com/kpot/keras-transformer.git
cd keras-transformer;pip3 install --user .
ln -s ~/simplicialtransformer/python/boxworld_v2.py ~/.local/lib/python3.6/site-packages/boxworld_v2.py
ln -s ~/simplicialtransformer/python/boxworld_v3.py ~/.local/lib/python3.6/site-packages/boxworld_v3.py
ln -s ~/simplicialtransformer/python/boxworld_v4.py ~/.local/lib/python3.6/site-packages/boxworld_v4.py
ln -s ~/simplicialtransformer/python/boxworld_v5.py ~/.local/lib/python3.6/site-packages/boxworld_v5.py
ln -s ~/simplicialtransformer/python/boxworld_agent_v1.py ~/.local/lib/python3.6/site-packages/boxworld_agent_v1.py
ln -s ~/simplicialtransformer/python/boxworld_agent_v2.py ~/.local/lib/python3.6/site-packages/boxworld_agent_v2.py
cp ~/simplicialtransformer/python/policy_evaluator-0.7.0.dev2-edited.py ~/.local/lib/python3.6/site-packages/ray/rllib/evaluation/policy_evaluator.py
cp ~/simplicialtransformer/python/impala-0.7.0.dev2-edited.py ~/.local/lib/python3.6/site-packages/ray/rllib/agents/impala/impala.py
cp ~/simplicialtransformer/python/pbt-0.7.0.dev2-edited.py ~/.local/lib/python3.6/site-packages/ray/tune/schedulers/pbt.py
```

Reboot after this to fix the PATH. You'll also need to open the port `6379` for Redis and the `8888` port for Jupyter in the console Security Groups tab, otherwise RLlib won't be able to initialise the cluster (resp. the Jupyter notebook will not be remotely accessible).

**Jupyter setup** (for head nodes only): To set up Jupyter as a remote service, follow [these instructions](https://jupyter-notebook.readthedocs.io/en/latest/public_server.html) (including making a keypair) *except* you need to use `c.NotebookApp.ip = '0.0.0.0'` rather than `c.NotebookApp.ip = '*'` as they say. To get Jupyter to run on startup you'll need to first create an `rc.local` file (on Ubuntu 18 this is no longer shipped standard) see [this](https://vpsfix.com/community/server-administration/no-etc-rc-local-file-on-ubuntu-18-04-heres-what-to-do/). Then add this line to `rc.local` (for Melbourne machines, otherwise use `murfetd` in place of `ubuntu`)

```
cd /home/ubuntu && su ubuntu -c "/home/ubuntu/.local/bin/jupyter notebook &"
```
