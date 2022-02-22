import numpy as np
import torch
import torch.nn as nn
import math

from torch.nn import functional as F

from pixelsnail import PixelSNAIL

class SLPModel(nn.Module):
	def __init__(self, args, ntoken, PADDING_INDEX):
		super(SLPModel, self).__init__()

		self.mt = MTModel(ntoken, PADDING_INDEX)

		self.cond_ch = 512

		self.hier = args.hier

		if args.hier == 'top':
			self.pixel = PixelSNAIL( [32, 32], 512, args.channel,
				5, 4, args.n_res_block, args.n_res_channel,
				n_cond_res_block=args.n_cond_res_block,
				cond_res_channel=args.n_res_channel,
				dropout=args.dropout, n_out_res_block=args.n_out_res_block,
			)

			self.downsample_conv = nn.Conv2d(self.cond_ch, self.cond_ch//2, 1)

		elif args.hier == 'bottom':
			self.pixel = PixelSNAIL( [64, 64], 512, args.channel,
				5, 4, args.n_res_block, args.n_res_channel, attention=False,
				dropout=args.dropout, n_cond_res_block=args.n_cond_res_block,
				cond_res_channel=args.n_res_channel,
			)

			self.merge_conv = nn.Conv2d(self.cond_ch*2, self.cond_ch, 1)

		self.pixel = nn.DataParallel(self.pixel)

	def forward(self, text, code, mask, condition=None):
		# INPUT IS BATCH FIRST, NEED TO TRANSFORM
		text = text.permute(1, 0)

		# print(f'text: {text.shape}')
		# print(f'mask: {mask.shape}')
		text_conditioning = self.mt(text, mask) # T x B x E
		# print(f'text_conditioning: {text_conditioning.shape}')
		T, B, E = text_conditioning.shape

		# WAY 1 FOR CONDITIONING
		# text_conditioning = text_conditioning.view(B*T, 32, 32) # T x (32 x 32) as B = 1
		# # print(f'text_conditioning after reshape: {text_conditioning.shape}')
		# text_conditioning = text_conditioning.unsqueeze(1)
		# text_conditioning = text_conditioning.repeat(1,self.cond_ch,1,1)

		# WAY 2 FOR CONDITIONING
		text_conditioning = text_conditioning.view(B*T, -1) # T x 512
		text_conditioning = text_conditioning.unsqueeze(-1).unsqueeze(-1) # T x 512 x 1 x 1
		text_conditioning = text_conditioning.repeat(1,1,32,32)
		
		if condition is not None:
			condition = (
                F.one_hot(condition, self.cond_ch)
                .permute(0, 3, 1, 2)
            )

			combined_condition = torch.cat((text_conditioning, condition), 1)

			text_conditioning = nn.ReLU()(self.merge_conv(combined_condition))
		else:
			text_conditioning = nn.ReLU()(self.downsample_conv(text_conditioning))
			
		# print(f'code: {code.shape} and text-conditioning: {text_conditioning.shape}')

		snail_output = self.pixel(code, condition=text_conditioning, text_conditioning=self.hier)[0] # T x (32 x 32)

		# print(f'snail_output: {snail_output.shape}')

		return snail_output

class MTModel(nn.Module):
	def __init__(self, ntoken, PADDING_INDEX):
		super(MTModel, self).__init__()

		self.d_model = 512
		nhead = 8
		dim_feedforward = 2048
		num_layers = 6
		dropout = 0.1
		activation = 'relu'

		self.encoder = nn.Embedding(ntoken, self.d_model)

		self.transformer_encoder = TransformerEncoderBundle(
			d_model=self.d_model, nhead=nhead, dim_feedforward=dim_feedforward,
			num_layers=num_layers, dropout=dropout, activation=activation)

	def forward(self, text, src_key_padding_mask):
		# Every operation is done batch first, using multiple GPUs.

		encodings = self.encoder(text) * math.sqrt(self.d_model)

		# print(f'encoding size: {encodings.shape}')

		encodings = self.transformer_encoder(encodings, src_key_padding_mask) # B X T X E

		return encodings

class TransformerEncoderBundle(nn.Module):
	def __init__(self, d_model, nhead, dim_feedforward, num_layers, dropout, activation):
		super(TransformerEncoderBundle, self).__init__()
		
		layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation) 

		norm = nn.LayerNorm(d_model)

		self.pos_encoder = PositionalEncoding(d_model, dropout)

		self.encoder = nn.TransformerEncoder(layers, num_layers, norm)

		self.init_weights()

	def init_weights(self):
		for p in self.parameters():
			if p.dim() > 1:
				nn.init.xavier_uniform_(p)

	def forward(self, x, src_key_padding_mask):
		x = self.pos_encoder(x)

		encodings = self.encoder(x, src_key_padding_mask=src_key_padding_mask)

		return encodings

class PositionalEncoding(nn.Module):
	def __init__(self, d_model, dropout=0.1, max_len=5000, batch_first=True):
		super(PositionalEncoding, self).__init__()
		self.dropout = nn.Dropout(p=dropout)
		self.batch_first = batch_first
		
		pe = torch.zeros(max_len, d_model)
		position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
		div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
		pe[:, 0::2] = torch.sin(position * div_term)
		pe[:, 1::2] = torch.cos(position * div_term)
		pe = pe.unsqueeze(0).transpose(0, 1)
		self.register_buffer('pe', pe)

	def forward(self, x):
		if self.batch_first:
			x = x.permute(1, 0, 2) # T X B X d_model
		
		x = x + self.pe[:x.size(0), :]
		
		if self.batch_first:
			x = x.permute(1, 0, 2) # B X T X d_model
		
		return self.dropout(x)