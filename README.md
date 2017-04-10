# rlcourse-april-7-christopher-beckham

This was the presentation I was meant to present in the last lecture of this course. It's an interesting paper that looks at using 
reinforcement learning (more specifically, Q-learning) to find good deep neural network architectures. It's motivated by the fact
that choosing a good (convolutional) network architecture involves a lot of experimentation and intuition. For example, some of these
hyperparameters include: the number of layers to use, the receptive field size, the number of feature maps per layer, nonlinearity,
and so forth.

In the paper, the authors frame the 'construction' of a convolutional architecture as an episode through a large MDP, where the states
consist of layers, and taking an action from state `s` to `s'` is a connection from layer `s` to layer `s'`. For example, the starting
layer -- the input layer -- will have many actions connecting it to different layers, perhaps a `conv(3,64)` (i.e., a convolutional layer
with filter size 3 and 64 output feature maps), or a `conv(5,32)`, or a `conv(5,128)`, and so forth. In order to limit the depth of the
network, each state has an associated layer depth parameter, and so the agent can only transition from a layer with depth `i` to depth `i+1`.
Other constraints are added to the MDP to keep the architecture reasonable tractable to train. For example, there can only be a maximum
of two fully-connected (FC) layers, and we cannot transition from a small FC layer to a bigger FC layer. An episode terminates when an
agent adds a softmax layer to the network (which can happen at any layer depth). The reward received at the goal state is the performance
on the validation set for that particular architecture. The layer types are kept basic: convolution, max-pool, and fully-connected are the
basic building blocks. The algorithm used is tabular Q-learning.

Note that this technique is not a complete end-to-end technique for identifying and training conv-nets using RL: the algorithm merely
selects good 'candidate' architectures, and from these candidates the authors re-train and fine-tune them. This is done for two reasons:
to keep the RL algorithm tractable (the 'reward' for an episode is simply the validation set on that architecture trained for 20 epochs,
and as we know, we need to train for hundreds of epochs to get good performance on image datasets); and because the algorithm doesn't 
consider training hyperparameters, such as optimiser, learning rate, batch size, and so forth. So we still have a long ways to go, but 
this is an excellent proof-of-concept paper and I think it is really neat.

The authors achieve state-of-the-art performance when comparing their candidate architectures to other architectures in the literature
that use the same basic building blocks.

In their work, the authors conclude that future work could entail using function approximation for the Q-learning instead of tabular.
Because they currently use tabular Q-learning, they have carefully discretised the state space so as to keep it a reasonable size. For
example, for the convolutional layer, the # of feature maps is discretised to {64,128,256,512}.
