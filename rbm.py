from __future__ import print_function
import numpy as np


class RBM:
    def __init__(self, num_visible, num_hidden):
        self.num_hidden = num_hidden
        self.num_visible = num_visible
        self.debug_print = True

        # 初始化一个权值矩阵，维数(num_visible , num_hidden)，使用
        # 在-sqrt(6。/ (num_hidden + num_visible))
        # 和sqrt(6。/ (num_hidden + num_visible))。可以改变
        # 用适当的值乘以区间的标准差。
        # 这里我们初始化权值，均值为0，标准差为0.1。
        # Reference:理解训练深度前馈的困难
        np_rng = np.random.RandomState(1234)

        self.weights = np.asarray(
            np_rng.uniform(
                low=-0.1 * np.sqrt(6.0 / (num_hidden + num_visible)),
                high=0.1 * np.sqrt(6.0 / (num_hidden + num_visible)),
                size=(num_visible, num_hidden),
            )
        )

        # 将偏置单位的权重插入第一行和第一列.
        self.weights = np.insert(self.weights, 0, 0, axis=0)
        self.weights = np.insert(self.weights, 0, 0, axis=1)

    def train(self, data, max_epochs=1000, learning_rate=0.1):
        """
        Train the machine.

        Parameters
        ----------
        data: 一个矩阵，其中每一行都是由可见单位的状态组成的训练示例.
        """

        num_examples = data.shape[0]

        # 将偏置单位1插入第一列.
        data = np.insert(data, 0, 1, axis=1)

        for epoch in range(max_epochs):
            # 夹住隐藏单位的数据和样本.
            # (这是“正CD阶段”，也就是现实阶段)
            pos_hidden_activations = np.dot(data, self.weights)
            pos_hidden_probs = self._logistic(pos_hidden_activations)
            pos_hidden_probs[:, 0] = 1  # Fix the bias unit.
            pos_hidden_states = pos_hidden_probs > np.random.rand(
                num_examples, self.num_hidden + 1
            )
            # Note that we're using the activation *probabilities* of the hidden states, not the hidden states
            # themselves, when computing associations. We could also use the states; see section 3 of Hinton's
            # "A Practical Guide to Training Restricted Boltzmann Machines" for more.
            pos_associations = np.dot(data.T, pos_hidden_probs)

            # 重建可见单位，并从隐藏单位再次采样.
            # (这是“消极CD阶段”，又名白日梦阶段.)
            neg_visible_activations = np.dot(pos_hidden_states, self.weights.T)
            neg_visible_probs = self._logistic(neg_visible_activations)
            neg_visible_probs[:, 0] = 1  # Fix the bias unit.
            neg_hidden_activations = np.dot(neg_visible_probs, self.weights)
            neg_hidden_probs = self._logistic(neg_hidden_activations)
            # Note, again, that we're using the activation *probabilities* when computing associations, not the states
            # themselves.
            neg_associations = np.dot(neg_visible_probs.T, neg_hidden_probs)

            # Update weights.
            self.weights += learning_rate * (
                (pos_associations - neg_associations) / num_examples
            )

            error = np.sum((data - neg_visible_probs) ** 2)
            if self.debug_print:
                print("Epoch %s: error is %s" % (epoch, error))

    def run_visible(self, data):
        """
        Assuming the RBM has been trained (so that weights for the network have been learned),
        run the network on a set of visible units, to get a sample of the hidden units.

        Parameters
        ----------
        data: A matrix where each row consists of the states of the visible units.

        Returns
        -------
        hidden_states: A matrix where each row consists of the hidden units activated from the visible
        units in the data matrix passed in.
        """

        num_examples = data.shape[0]

        # Create a matrix, where each row is to be the hidden units (plus a bias unit)
        # sampled from a training example.
        hidden_states = np.ones((num_examples, self.num_hidden + 1))

        # Insert bias units of 1 into the first column of data.
        data = np.insert(data, 0, 1, axis=1)

        # Calculate the activations of the hidden units.
        hidden_activations = np.dot(data, self.weights)
        # Calculate the probabilities of turning the hidden units on.
        hidden_probs = self._logistic(hidden_activations)
        # Turn the hidden units on with their specified probabilities.
        hidden_states[:, :] = hidden_probs > np.random.rand(
            num_examples, self.num_hidden + 1
        )
        # Always fix the bias unit to 1.
        # hidden_states[:,0] = 1

        # Ignore the bias units.
        hidden_states = hidden_states[:, 1:]
        return hidden_states

    # TODO: Remove the code duplication between this method and `run_visible`?
    def run_hidden(self, data):
        """
        Assuming the RBM has been trained (so that weights for the network have been learned),
        run the network on a set of hidden units, to get a sample of the visible units.

        Parameters
        ----------
        data: A matrix where each row consists of the states of the hidden units.

        Returns
        -------
        visible_states: A matrix where each row consists of the visible units activated from the hidden
        units in the data matrix passed in.
        """

        num_examples = data.shape[0]

        # Create a matrix, where each row is to be the visible units (plus a bias unit)
        # sampled from a training example.
        visible_states = np.ones((num_examples, self.num_visible + 1))

        # Insert bias units of 1 into the first column of data.
        data = np.insert(data, 0, 1, axis=1)

        # Calculate the activations of the visible units.
        visible_activations = np.dot(data, self.weights.T)
        # Calculate the probabilities of turning the visible units on.
        visible_probs = self._logistic(visible_activations)
        # Turn the visible units on with their specified probabilities.
        visible_states[:, :] = visible_probs > np.random.rand(
            num_examples, self.num_visible + 1
        )
        # Always fix the bias unit to 1.
        # visible_states[:,0] = 1

        # Ignore the bias units.
        visible_states = visible_states[:, 1:]
        return visible_states

    def daydream(self, num_samples):
        """
        随机初始化可见单位一次，并开始运行交替吉布斯采样步骤
        (每一步都包含更新所有隐藏单元，然后更新所有可见单元)，
        在每个步骤中抽取可见单元的样本。
        注意，我们只初始化网络*一次*，所以这些样本是相关的。

        Returns
        -------
        samples: A matrix, where each row is a sample of the visible units produced while the network was
        daydreaming.
        """

        # Create a matrix, where each row is to be a sample of of the visible units
        # (with an extra bias unit), initialized to all ones.
        samples = np.ones((num_samples, self.num_visible + 1))

        # Take the first sample from a uniform distribution.
        samples[0, 1:] = np.random.rand(self.num_visible)

        # Start the alternating Gibbs sampling.
        # Note that we keep the hidden units binary states, but leave the
        # visible units as real probabilities. See section 3 of Hinton's
        # "A Practical Guide to Training Restricted Boltzmann Machines"
        # for more on why.
        for i in range(1, num_samples):
            visible = samples[i - 1, :]

            # Calculate the activations of the hidden units.
            hidden_activations = np.dot(visible, self.weights)
            # Calculate the probabilities of turning the hidden units on.
            hidden_probs = self._logistic(hidden_activations)
            # Turn the hidden units on with their specified probabilities.
            hidden_states = hidden_probs > np.random.rand(self.num_hidden + 1)
            # Always fix the bias unit to 1.
            hidden_states[0] = 1

            # Recalculate the probabilities that the visible units are on.
            visible_activations = np.dot(hidden_states, self.weights.T)
            visible_probs = self._logistic(visible_activations)
            visible_states = visible_probs > np.random.rand(self.num_visible + 1)
            samples[i, :] = visible_states

        # Ignore the bias units (the first column), since they're always set to 1.
        return samples[:, 1:]

    def _logistic(self, x):
        return 1.0 / (1 + np.exp(-x))


if __name__ == "__main__":
    r = RBM(num_visible=6, num_hidden=2)
    training_data = np.array(
        [
            [1, 1, 1, 0, 0, 0],
            [1, 0, 1, 0, 0, 0],
            [1, 1, 1, 0, 0, 0],
            [0, 0, 1, 1, 1, 0],
            [0, 0, 1, 1, 0, 0],
            [0, 0, 1, 1, 1, 0],
        ]
    )
    r.train(training_data, max_epochs=5000)
    print(r.weights)
    user = np.array([[0, 0, 0, 1, 1, 0]])
    print(r.run_visible(user))
