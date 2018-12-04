import torch
import torch.nn as nn
import torch.nn.functional as F

CONTEXT_SIZE = 2  # 2 words to the left, 2 to the right
MAX_EPOCH = 100
raw_text = """We are about to study the idea of a computational process.
Computational processes are abstract beings that inhabit computers.
As they evolve, processes manipulate other abstract things called data.
The evolution of a process is directed by a pattern of rules
called a program. People create programs to direct processes. In effect,
we conjure the spirits of the computer with our spells.""".split()

# By deriving a set from `raw_text`, we deduplicate the array
vocab = set(raw_text)
vocab_size = len(vocab)

word_to_ix = {word: i for i, word in enumerate(vocab)}
data = []
for i in range(2, len(raw_text) - 2):
    context = [raw_text[i - 2], raw_text[i - 1],
               raw_text[i + 1], raw_text[i + 2]]
    target = raw_text[i]
    data.append((context, target))
print(data[:5])


def make_context_vector(context, word_to_ix):
    idxs = [word_to_ix[w] for w in context]
    return torch.tensor(idxs, dtype=torch.long)


class CBOW(nn.Module):

    def __init__(self, embedding_dim):
        super(CBOW, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, inputs):
        out = self.embedding(inputs).sum(dim=0)
        out = self.linear(out)
        log_prob = F.log_softmax(out, dim=0).unsqueeze(0)
        return log_prob


# create your model and train.  here are some functions to help you make
# the data ready for use by your module
model = CBOW(10)
loss = nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(), 0.0001, momentum=0.9)
for epoch in range(MAX_EPOCH):
    for context, target in data:
        input = make_context_vector(context, word_to_ix)
        output = model(input)
        loss_val = loss(output, torch.tensor([word_to_ix[target]], dtype=torch.long))
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()
        print(loss_val.item())
