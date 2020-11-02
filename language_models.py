import torch.nn as nn
import torch
import torch.nn.functional as F


class UniRNN(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, target_size):
        super(uniRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, batch_first = True)
        self.output_layer = nn.Linear(hidden_size, target_size)
        

    def forward(self, sentence_batch):
        # sentence_batch = Variable(sentence_batch)
        
        token_embedding = self.embedding(sentence_batch)
        token_embedding = F.tanh(token_embedding)

        lstm_out, _ = self.lstm(token_embedding)
        lstm_out = lstm_out.contiguous()
        lstm_out = lstm_out[:,-1,:]
        
        logits = self.output_layer(lstm_out)

        return logits

class CnnTextClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, target_size, window_sizes=(3, 4, 5)):
        super(CnnTextClassifier, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_size)

        self.convs = nn.ModuleList([
            nn.Conv2d(1, hidden_size, [window_size, embedding_size], padding=(window_size - 1, 0))
            for window_size in window_sizes
        ])

        self.fc = nn.Linear(hidden_size * len(window_sizes), target_size)

    def forward(self, sentence_batch):
        
        token_embedding = self.embedding(sentence_batch)

        # Apply a convolution + max pool layer for each window size
        x = torch.unsqueeze(token_embedding, 1)       # [B, C, T, E] Add a channel dim.
        xs = []
        for conv in self.convs:
            x2 = F.relu(conv(x))        # [B, F, T, 1]
            x2 = torch.squeeze(x2, -1)  # [B, F, T]
            x2 = F.max_pool1d(x2, x2.size(2))  # [B, F, 1]
            xs.append(x2)
        x = torch.cat(xs, 2)            # [B, F, window]

        # FC
        x = x.view(x.size(0), -1)       # [B, F * window]
        logits = self.fc(x)             # [B, class]

        return logits

def get_model(model_type, vocab_size, embedding_size, hidden_size, target_size, cuda = True):
	assert model_type in ["uni_rnn", "cnn"]
	if model_type == "uni_rnn":
		model = UniRNN(vocab_size, embedding_size, hidden_size, target_size)
	elif model_type == "cnn":
		model = CnnTextClassifier(vocab_size, embedding_size, hidden_size, target_size)
	else:
		raise Exception("Not Implemented")
	
	if cuda:
		model = model.cuda()

	return model

