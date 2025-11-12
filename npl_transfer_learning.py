import torch
import torchtext
from tqdm import tqdm
import numpy as np
import spacy
from transformers import BertTokenizer
from transformers import BertModel

nlp = spacy.load("en_core_web_sm")

TEXT = torchtext.data.Field(tokenize="spacy")
LABEL = torchtext.data.LabelField(dtype=torch.long)

train_data, test_data = torchtext.datasets.IMDB.splits(TEXT, LABEL)

print(len(train_data), len(test_data))

print(vars(train_data.examples[0]))


MAX_VOCAB_SIZE = 10000

TEXT.build_vocab(
    train_data,
    max_size=MAX_VOCAB_SIZE,
    vectors="glove.6B.100d",  # embeddings pre-entrenados
    unk_init=torch.Tensor.normal_,
)

LABEL.build_vocab(train_data)

print(len(TEXT.vocab), len(LABEL.vocab))

device = "cuda" if torch.cuda.is_available() else "cpu"

dataloader = {
    "train": torchtext.data.BucketIterator(
        train_data, batch_size=64, shuffle=True, sort_within_batch=True, device=device
    ),
    "test": torchtext.data.BucketIterator(test_data, batch_size=64, device=device),
}


class RNN(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        embedding_dim=128,
        hidden_dim=128,
        output_dim=2,
        num_layers=2,
        dropout=0.2,
        bidirectional=False,
    ):
        super().__init__()
        self.embedding = torch.nn.Embedding(input_dim, embedding_dim)
        self.rnn = torch.nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )
        self.fc = torch.nn.Linear(
            2 * hidden_dim if bidirectional else hidden_dim, output_dim
        )

    def forward(self, text):
        # no entrenamos los embeddings
        with torch.no_grad():
            # text = [sent len, batch size]
            embedded = self.embedding(text)
        # embedded = [sent len, batch size, emb dim]
        output, hidden = self.rnn(embedded)
        # output = [sent len, batch size, hid dim]
        y = self.fc(output[-1, :, :].squeeze(0))
        return y


model = RNN(input_dim=len(TEXT.vocab), bidirectional=True, embedding_dim=100)

pretrained_embeddings = TEXT.vocab.vectors
model.embedding.weight.data.copy_(pretrained_embeddings)
# ponemos a cero los pesos correspondientes a los tokens <unk> y <pad>
model.embedding.weight.data[TEXT.vocab.stoi[TEXT.unk_token]] = torch.zeros(100)
model.embedding.weight.data[TEXT.vocab.stoi[TEXT.pad_token]] = torch.zeros(100)

outputs = model(torch.randint(0, len(TEXT.vocab), (100, 64)))
print(outputs.shape)


def fit(model, dataloader, epochs=5):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, train_acc = [], []
        bar = tqdm(dataloader["train"])
        for batch in bar:
            X, y = batch
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            y_hat = model(X)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            acc = (y == torch.argmax(y_hat, axis=1)).sum().item() / len(y)
            train_acc.append(acc)
            bar.set_description(
                f"loss {np.mean(train_loss):.5f} acc {np.mean(train_acc):.5f}"
            )
        bar = tqdm(dataloader["test"])
        val_loss, val_acc = [], []
        model.eval()
        with torch.no_grad():
            for batch in bar:
                X, y = batch
                X, y = X.to(device), y.to(device)
                y_hat = model(X)
                loss = criterion(y_hat, y)
                val_loss.append(loss.item())
                acc = (y == torch.argmax(y_hat, axis=1)).sum().item() / len(y)
                val_acc.append(acc)
                bar.set_description(
                    f"val_loss {np.mean(val_loss):.5f} val_acc {np.mean(val_acc):.5f}"
                )
        print(
            f"Epoch {epoch}/{epochs} loss {np.mean(train_loss):.5f} val_loss {np.mean(val_loss):.5f} acc {np.mean(train_acc):.5f} val_acc {np.mean(val_acc):.5f}"
        )


fit(model, dataloader)


def predict(model, X):
    model.eval()
    with torch.no_grad():
        X = torch.tensor(X).to(device)
        pred = model(X)
        return pred


sentences = [
    "this film is terrible",
    "this film is great",
    "this film is good",
    "a waste of time",
]
tokenized = [[tok.text for tok in nlp.tokenizer(sentence)] for sentence in sentences]
indexed = [[TEXT.vocab.stoi[_t] for _t in t] for t in tokenized]
tensor = torch.tensor(indexed).permute(1, 0)
predictions = torch.argmax(predict(model, tensor), axis=1)
print(predictions)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

tokens = tokenizer.tokenize("Hello WORLD how ARE yoU?")
print(tokens)

indexes = tokenizer.convert_tokens_to_ids(tokens)
print(indexes)

max_input_length = tokenizer.max_model_input_sizes["bert-base-uncased"]


def tokenize_and_cut(sentence):
    tokens = tokenizer.tokenize(sentence)
    tokens = tokens[: max_input_length - 2]
    return tokens


TEXT = torchtext.data.Field(
    batch_first=True,
    use_vocab=False,
    tokenize=tokenize_and_cut,
    preprocessing=tokenizer.convert_tokens_to_ids,
    init_token=tokenizer.cls_token_id,
    eos_token=tokenizer.sep_token_id,
    pad_token=tokenizer.pad_token_id,
    unk_token=tokenizer.unk_token_id,
)

LABEL = torchtext.data.LabelField(dtype=torch.long)

train_data, test_data = torchtext.datasets.IMDB.splits(TEXT, LABEL)

LABEL.build_vocab(train_data)

dataloader = {
    "train": torchtext.data.BucketIterator(
        train_data, batch_size=64, shuffle=True, sort_within_batch=True, device=device
    ),
    "test": torchtext.data.BucketIterator(test_data, batch_size=64, device=device),
}


class BERT(torch.nn.Module):
    def __init__(
        self, hidden_dim=256, output_dim=2, n_layers=2, bidirectional=True, dropout=0.2
    ):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")

        # freeze BERT
        for name, param in self.bert.named_parameters():
            if name.startswith("bert"):
                param.requires_grad = False

        embedding_dim = self.bert.config.to_dict()["hidden_size"]
        self.rnn = torch.nn.GRU(
            embedding_dim,
            hidden_dim,
            num_layers=n_layers,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=0 if n_layers < 2 else dropout,
        )

        self.fc = torch.nn.Linear(
            hidden_dim * 2 if bidirectional else hidden_dim, output_dim
        )

    def forward(self, text):
        with torch.no_grad():
            embedded = self.bert(text)[0]
        output, hidden = self.rnn(embedded)
        y = self.fc(output[:, -1, :].squeeze(1))
        return y


model = BERT()
fit(model, dataloader, epochs=3)


def predict(sentence):
    tokenized = [tok[: max_input_length - 2] for tok in tokenizer.tokenize(sentence)]
    indexed = (
        [tokenizer.cls_token_id]
        + tokenizer.convert_tokens_to_ids(tokenized)
        + [tokenizer.sep_token_id]
    )
    tensor = torch.tensor([indexed]).to(device)
    model.eval()
    return torch.argmax(model(tensor), axis=1)


sentences = ["Best film ever !", "this movie is terrible"]
preds = [predict(s) for s in sentences]
print(preds)
