import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt


class LinearBBB(nn.Module):
    """
    Layer of BBB
    """

    def __init__(self, input_features, output_features, prior_var=1.):
        """
        先验分布是以均值为0，方差为1的高斯分布
        """
        super().__init__()
        # input and output dimension
        self.input_features = input_features
        self.output_features = output_features

        # initialize parameters for weights of the layer
        # 初始化该层的去哪找和偏置==》y = w*x+b
        # 该层的每一个权重和偏置都有自己的方差和均值
        self.w_mu = nn.Parameter(torch.zeros(output_features, input_features))
        self.w_rho = nn.Parameter(torch.zeros(output_features, input_features))

        # initialize parameters for the layer's bias
        self.b_mu = nn.Parameter(torch.zeros(output_features))
        self.b_rho = nn.Parameter(torch.zeros(output_features))

        # initialize weight samples
        self.w = None
        self.b = None

        # 假设该层的所有权重和偏置都是正态分布
        self.prior = torch.distributions.Normal(0, prior_var)

    def forward(self, inputs):
        """
          Optimization process
        """
        # sample weights
        """
            从均值为0，方差为1的高斯分布中采样一些样本点 u + log(1 + exp(p)) * w'
            一种重参数技巧，最早用于VAE中
            对于原本服从 N~（u , p）的随机变量w，先不直接根据这个分布采样，而是先根据标准正态分布采样 w_epsilon
            随后根据 w = u + log(1 + exp(p)) * w' 得到实际的采样值，这样做的目的是为了便于反向传播
        """

        w_epsilon = Normal(0, 1).sample(self.w_mu.shape)
        self.w = self.w_mu + torch.log(1 + torch.exp(self.w_rho)) * w_epsilon

        # sample bias
        b_epsilon = Normal(0, 1).sample(self.b_mu.shape)
        self.b = self.b_mu + torch.log(1 + torch.exp(self.b_rho)) * b_epsilon

        # record log prior by evaluating log pdf of prior at sampled weight and bias
        """
            对已经采样的值，计算其在预先定义的分布上的对数形式得到的值（log_prob(value)是计算value在定义的正态分布（mean,1）中对应的概率的对数）
            损失函数是要最大化elbo下界：L = sum[log(q(w))]- sum(log P(w)) - sum(log P(y_i | w, x_i))
        """
        # 计算 p（w）
        w_log_prior = self.prior.log_prob(self.w)
        b_log_prior = self.prior.log_prob(self.b)
        self.log_prior = torch.sum(w_log_prior) + torch.sum(b_log_prior)

        # record log variational posterior by evaluating log pdf of normal distribution defined by parameters with
        # respect at the sampled values
        # 计算 q(w)，也有称其为 p(w|theta)的 q(w) 表示根据当前的网络参数 w_mu，w_rho，b_mu，b_rho
        # 计算q（w），也就是说q（w）就是我们损失函数要求解的分布
        self.w_post = Normal(self.w_mu.data, torch.log(1 + torch.exp(self.w_rho)))
        self.b_post = Normal(self.b_mu.data, torch.log(1 + torch.exp(self.b_rho)))
        self.log_post = self.w_post.log_prob(self.w).sum() + self.b_post.log_prob(self.b).sum()

        return F.linear(inputs, self.w, self.b)


class MLP_BBB(nn.Module):
    def __init__(self, hidden_units, noise_tol=.1, prior_var=1.):
        # initialize the network like you would with a standard multilayer perceptron, but using the BBB layer
        super().__init__()
        self.hidden = LinearBBB(1, hidden_units, prior_var=prior_var)
        self.out = LinearBBB(hidden_units, 1, prior_var=prior_var)
        self.noise_tol = noise_tol  # we will use the noise tolerance to calculate our likelihood

    def forward(self, x):
        # again, this is equivalent to a standard multilayer perceptron
        x = torch.sigmoid(self.hidden(x))
        x = self.out(x)
        return x

    def log_prior(self):
        # calculate the log prior over all the layers
        return self.hidden.log_prior + self.out.log_prior

    def log_post(self):
        # calculate the log posterior over all the layers
        return self.hidden.log_post + self.out.log_post

    # 损失函数的计算
    def sample_elbo(self, input, target, samples):
        # we calculate the negative elbo, which will be our loss function
        # initialize tensors
        outputs = torch.zeros(samples, target.shape[0])
        log_priors = torch.zeros(samples)
        log_posts = torch.zeros(samples)
        log_likes = torch.zeros(samples)
        # make predictions and calculate prior, posterior, and likelihood for a given number of samples
        # 蒙特卡洛近似，根据给定的样本数采样
        # 蒙特卡洛在这里用来计算前向传播的次数，所以蒙特卡洛可能与模型权重的不确定性有关
        for i in range(samples):
            outputs[i] = self(input).reshape(-1)  # make predictions
            log_priors[i] = self.log_prior()  # get log prior
            log_posts[i] = self.log_post()  # get log variational posterior
            log_likes[i] = Normal(outputs[i], self.noise_tol).log_prob(
                target.reshape(-1)).sum()  # calculate the log likelihood
        # calculate monte carlo estimate of prior posterior and likelihood
        log_prior = log_priors.mean()
        log_post = log_posts.mean()
        log_like = log_likes.mean()
        # calculate the negative elbo (which is our loss function)
        loss = log_post - log_prior - log_like
        return loss


def toy_function(x):
    return -x ** 4 + 3 * x ** 2 + 1


# toy dataset we can start with
x = torch.tensor([-2, -1.8, -1, 1, 1.8, 2]).reshape(-1, 1)
y = toy_function(x)

net = MLP_BBB(32, prior_var=10)
optimizer = optim.Adam(net.parameters(), lr=.1)
epochs = 1000
for epoch in range(epochs):  # loop over the dataset multiple times
    optimizer.zero_grad()
    # forward + backward + optimize
    loss = net.sample_elbo(x, y, 1)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print('epoch: {}/{}'.format(epoch + 1, epochs))
        print('Loss:', loss.item())
print('Finished Training')

# samples is the number of "predictions" we make for 1 x-value.
# 网络训练完成后，对于随机给定的输入值 x，计算模型的输出值 y
# samples 表示采样次数，即预测多少次，最后对这些次的结果取平均值
samples = 10
x_tmp = torch.linspace(-5, 5, 100).reshape(-1, 1)
y_samp = np.zeros((samples, 100))
print(x_tmp.shape)
for s in range(samples):
    y_tmp = net(x_tmp).detach().numpy()
    y_samp[s] = y_tmp.reshape(-1)
plt.plot(x_tmp.numpy(), np.mean(y_samp, axis=0), label='Mean Posterior Predictive')
plt.fill_between(x_tmp.numpy().reshape(-1), np.percentile(y_samp, 2.5, axis=0),
                 np.percentile(y_samp, 97.5, axis=0), alpha=0.25, label='95% Confidence')
plt.legend()
plt.scatter(x, toy_function(x))
plt.title('Posterior Predictive')
plt.show()

if __name__ == '__main__':
    print("done!")


