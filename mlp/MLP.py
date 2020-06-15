class MLP:
    def __init__(self):
        input_size = 4
        hidden_size = 3
        output_size = 3
        self.hist = {'loss':[], 'acc':[]}
        
#         self.W1 = 
#         self.b1 = 
#         self.W2 = 
#         self.b2 = 

    def softmax(self, x):
        pass
    
    def sigmoid(self, x):
        pass

    def cross_entropy(self, y, o):
        pass
    
    def forward(self, x):
#         h1 = sigmoid(x*W_1+b_1)
#         o = softmax(h1*W_2 + b_2)
#         return o
        pass

    def backward(self, y, o):
        pass
    
    def train(self, x, y, epochs):
        for epoch in tqdm(range(1, epochs+1)):
            o = self.forward(x)
            self.backward(y, o)
            
            loss = self.cross_entropy(y, o)
            acc = accuracy_score(y, o)
            self.hist['loss'] += [loss]
            self.hist['acc'] += [acc]
            print(epoch, 'loss:', loss, 'acc:', acc)