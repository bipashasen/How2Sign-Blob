import numpy as np
import torch
import torch.nn as nn
import math

from tqdm import tqdm

from torch.nn import functional as F

from pixelcnn.models import GatedPixelCNN

class SLPModel(nn.Module):
	def __init__(self, ntoken, n_class_vq, img_dim, n_layers_pixelcnn, batch_size):
		super(SLPModel, self).__init__()

		self.mt = MTModel(ntoken, img_dim, batch_size)

		self.pixel_cnn = nn.DataParallel(GatedPixelCNN(
			n_class_vq, img_dim**2, n_layers_pixelcnn))

		self.init_weights()

	def init_weights(self):
		for p in self.parameters():
			if p.dim() > 1:
				nn.init.xavier_uniform_(p)

	def forward(self, code, text, mask):
		text_encodings = self.mt(text, mask).squeeze(1) # T x E (B = 1)

		outputs = self.pixel_cnn(code, text_encodings)

		return outputs

	def generate(self, x, text, mask):
		shape = x.shape

		text_encodings = self.mt(text, mask).squeeze(1)

		print('starting generation')
		for i in tqdm(range(1)):
			for j in range(1):
				logits = self.pixel_cnn(x, text_encodings)
				probs = F.softmax(logits[:, :, i, j], -1)
				
				x.data[:, i, j].copy_(
					probs.multinomial(1).squeeze().data
				)
		return x

class MTModel(nn.Module):
	def __init__(self, ntoken, img_shape, out_seq_len):
		super(MTModel, self).__init__()

		self.d_model = (img_shape**2) * 2
		nhead = 5
		dim_feedforward = 2048
		num_layers = 6
		dropout = 0.1
		activation = 'relu'

		self.encoder = nn.Embedding(ntoken, self.d_model)

		self.transformer_encoder = TransformerEncoderBundle(
			d_model=self.d_model, nhead=nhead, dim_feedforward=dim_feedforward,
			num_layers=num_layers, dropout=dropout, activation=activation)

		self.gru_decoder = MTDecoder(
			d_model=self.d_model, num_layers=3, seq_len=out_seq_len)

	def forward(self, text, src_key_padding_mask):
		encodings = self.encoder(text) * math.sqrt(self.d_model)

		encodings = encodings.permute(1, 0, 2)
		
		encodings = self.transformer_encoder(encodings, src_key_padding_mask) # T x B x D
		
		return self.gru_decoder(encodings)

class MTDecoder(nn.Module):
	def __init__(self, d_model, num_layers, seq_len):
		super(MTDecoder, self).__init__()

		self.num_layers = num_layers

		self.gru = nn.GRU(
			input_size=d_model, 
			hidden_size=d_model, 
			num_layers=num_layers)

		self.attention = nn.Sequential(
			nn.Linear(d_model*2, d_model),
			nn.ReLU(inplace=True),
			nn.Linear(d_model, d_model//4),
			nn.ReLU(inplace=True),
			nn.Linear(d_model//4, 1)
		)

		self.transform = nn.Sequential(
			nn.Linear(d_model*2, d_model),
			nn.ReLU(inplace=True),
			nn.Linear(d_model, d_model),
			nn.ReLU(inplace=True),
		)

		self.seq_len = seq_len

	def cross_attention(self, enc_x, y):
		T, B, D = enc_x.shape # T x B x 625 (B = 1)

		embed = y.repeat(T, 1, 1) # 1 x B x 625 -> T x B x 625
		combined = torch.cat((embed, enc_x), axis = 2) # T x 1 x 1025
		combined = combined.reshape(-1, D*2) # T x 1025
		
		attention = self.attention(combined) # T
		attention = attention.view(T, B, 1) # T x 1 x 1
		attention = nn.Softmax(dim=0)(attention) # T x B x 1

		context = (attention * enc_x).sum(0).unsqueeze(0) # 1 x B x 625

		combined = torch.cat((y, context), axis = 2) # T x B x 1025
		
		return self.transform(combined) # 1 x B x 625

	def forward(self, enc_x):
		y = torch.zeros_like(enc_x[0]).unsqueeze(0)
		
		decodings = []
		out = enc_x[-1, :, :].repeat(self.num_layers, 1, 1)
		
		for i in range(self.seq_len):
			y = self.cross_attention(enc_x, y) # input: 1 x B x 625, 3 x B x 625

			y, out = self.gru(y, out)

			decodings.append(y)

		return torch.vstack(decodings)

class TransformerEncoderBundle(nn.Module):
	def __init__(self, d_model, nhead, dim_feedforward, num_layers, dropout, activation):
		super(TransformerEncoderBundle, self).__init__()
		
		layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation) 

		norm = nn.LayerNorm(d_model)

		self.pos_encoder = PositionalEncoding(d_model, dropout)

		self.encoder = nn.TransformerEncoder(layers, num_layers, norm)

	def forward(self, x, src_key_padding_mask):
		x = self.pos_encoder(x)
		
		return self.encoder(x, src_key_padding_mask=src_key_padding_mask)

class PositionalEncoding(nn.Module):
	def __init__(self, d_model, dropout=0.1, max_len=5000):
		super(PositionalEncoding, self).__init__()
		self.dropout = nn.Dropout(p=dropout)
		
		pe = torch.zeros(max_len, d_model)
		position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
		div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
		pe[:, 0::2] = torch.sin(position * div_term)
		pe[:, 1::2] = torch.cos(position * div_term)[:, :pe.shape[1]//2]
		pe = pe.unsqueeze(0).transpose(0, 1)
		self.register_buffer('pe', pe)

	def forward(self, x):
		x = x + self.pe[:x.size(0), :]
		
		return self.dropout(x)