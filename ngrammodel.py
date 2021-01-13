import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle as pckl

def make_context_vector(context, word_to_ix):
    idxs = [word_to_ix[w] for w in context]
    return torch.tensor(idxs, dtype=torch.long)

class NGramLanguageModeler(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_size):
        super(NGramLanguageModeler, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out)
        return log_probs


class CBOW(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, word_to_ix):
        super(CBOW, self).__init__()
        self.word_to_ix = word_to_ix

        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(embedding_dim, 128)
        self.activation_function1 = nn.ReLU()

        self.linear2 = nn.Linear(128, vocab_size)
        self.activation_function2 = nn.LogSoftmax(dim=-1)

    def forward(self, inputs):
        embeds = sum(self.embeddings(inputs)).view(1, -1)
        out = self.linear1(embeds)
        out = self.activation_function1(out)
        out = self.linear2(out)
        out = self.activation_function2(out)
        return out

    def get_word_emdedding(self, word):
        word = torch.tensor([self.word_to_ix[word]])
        return self.embeddings(word).view(1, -1)


class Embeddings:
    def __init__(self):
        self.losses = []
        self.loss_function = nn.NLLLoss()
        self.epochs = 2

    def trainEmbeddings(self, word_to_ix, ix_to_word, vocab_size, embeddding_dim, context_size, ngram):
        model = NGramLanguageModeler(vocab_size, embeddding_dim, context_size)
        optimizer = optim.SGD(model.parameters(), lr=0.001)

        for epoch in range(self.epochs):
            total_loss = 0
            for context, target in ngram:
                # Step 1. Prepare the inputs to be passed to the model (i.e, turn the words
                # into integer indices and wrap them in tensors)
                context_idxs = torch.tensor([word_to_ix[w] for w in context], dtype=torch.long)

                # Step 2. Recall that torch *accumulates* gradients. Before passing in a
                # new instance, you need to zero out the gradients from the old
                # instance
                model.zero_grad()

                # Step 3. Run the forward pass, getting log probabilities over next
                # words
                log_probs = model(context_idxs)

                # Step 4. Compute your loss function. (Again, Torch wants the target
                # word wrapped in a tensor)
                loss = self.loss_function(log_probs, torch.tensor([word_to_ix[target]], dtype=torch.long))

                # Step 5. Do the backward pass and update the gradient
                loss.backward()
                optimizer.step()

                # Get the Python number from a 1-element Tensor by calling tensor.item()
                total_loss += loss.item()
            self.losses.append(total_loss)
        print(self.losses)  # The loss decreased every iteration over the training data!
        self.serializeModel(model)

    def trainEmbeddingCBOW(self, ngram, embedding_dim, vocab_size, word_to_ix, ix_to_word, mode):

        if(mode == "GPU" and torch.cuda.is_available()):
            device = "cuda:0"
        else:
            device = "cpu"

        self.loss_function = nn.NLLLoss()
        model = CBOW(vocab_size, embedding_dim, word_to_ix)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

        print("Started training")

        for epoch in range(self.epochs):
            total_loss = 0
            for context, target in ngram:
                context_vector = make_context_vector(context, word_to_ix)
                model.zero_grad()

                log_probs = model(context_vector)

                loss = self.loss_function(log_probs, torch.tensor([word_to_ix[target]]))

                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            optimizer.zero_grad()
            self.losses.append(total_loss)
            print(epoch)

        print(self.losses)
        self.serializeModel(model)


    def serializeModel(self, model):
        with open('./models/embeddings.pickle', "wb") as file:
            pckl.dump(model, file)

    def loadModel(self):
        with open('./models/embeddings.pickle', "rb") as file:
            return pckl.load(file)

    def testModel(self, data, expected, word_to_ix, ix_to_word):
        print("Starting to test model")
        model = self.loadModel()
        for context, target in data:
            context_vector = make_context_vector(context, word_to_ix)
            answer = model(context_vector)
            print(f'Context: {context}\n')
            print(f'Prediction: {ix_to_word[torch.argmax(answer[0]).item()]}')
