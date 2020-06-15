class MLP:
    def __init__(self):
        input_size = 4
        hidden_size = 3
        output_size = 3
        self.hist = {'loss':[], 'acc':[]}
        self.lr = 1e-2
        
        self.W1 = torch.randn(input_size, hidden_size)
        self.b1 = torch.randn(hidden_size)
        self.W2 = torch.randn(hidden_size, output_size)
        self.b2 = torch.randn(output_size)

    def softmax(self, x):
        e = torch.exp(x - torch.max(x))
        return e / e.sum()
    
    def sigmoid(self, x):
        return 1 / (1 + torch.exp(x))

    def cross_entropy(self, y, o):
      return -torch.sum(y * torch.log(o + 1e-10)).item()
    
    def forward(self, x):
        self.h = self.sigmoid(torch.matmul(x, self.W1) + self.b1)
        o = self.softmax(torch.matmul(self.h, self.W2) + self.b2)
        return o

    def backward(self, y, o):
        dZ3 = o - y
        dW2 = torch.matmul(self.h.T, dZ3)
        db2 = torch.mean(dZ3, axis=0)
        dh = torch.matmul(dZ3, self.W2.T)
        dZ2 = dh * self.sigmoid(self.h) * (1 - self.sigmoid(self.h))
        dW1 = torch.matmul(self.x.T, dZ2)
        db1 = torch.mean(dZ2, axis=0)
        
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
    
    def train(self, x, y, epochs):
        self.x = x
        for epoch in tqdm(range(1, epochs+1)):
            o = self.forward(x)
            self.backward(y, o)
            
            loss = self.cross_entropy(y, o)
            acc = accuracy_score(np.argmax(y.numpy(), axis=1), np.argmax(o.numpy(), axis=1))
            print(np.argmax(y.numpy(), axis=1))
            print(np.argmax(o.numpy(), axis=1))
            self.hist['loss'] += [loss]
            self.hist['acc'] += [acc]
            print(epoch, 'loss:', loss, 'acc:', acc)