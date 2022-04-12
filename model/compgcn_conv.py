from helper import *
from model.message_passing import MessagePassing
from torch.nn import Sequential as seq, Parameter,LeakyReLU,init,Linear
from torch_geometric.utils import softmax,degree
import sys

class CompGCNConv(MessagePassing):
	def __init__(self, curv, in_channels, out_channels, num_rels, act=lambda x:x, params=None):
		super(self.__class__, self).__init__()

		self.p 			= params
		self.in_channels	= in_channels
		self.out_channels	= out_channels
		self.num_rels 		= num_rels
		self.act 		= act
		self.device		= None

		self.w_loop		= get_param((in_channels, out_channels))
		self.w_in		= get_param((in_channels, out_channels))
		self.w_out		= get_param((in_channels, out_channels))
		self.w_rel 		= get_param((in_channels, out_channels))
		self.loop_rel 		= get_param((1, in_channels))

		self.drop		= torch.nn.Dropout(self.p.dropout)
		self.bn			= torch.nn.BatchNorm1d(out_channels)
		
		self.curvy = curv.view(-1,1)
		widths=[1,out_channels]
		self.w_mlp_out=create_wmlp(widths,self.out_channels,1)

		if self.p.bias: self.register_parameter('bias', Parameter(torch.zeros(out_channels)))

	def forward(self, x, edge_index, edge_type, rel_embed): 
		if self.device is None:
			self.device = edge_index.device
			
		self.curv = self.w_mlp_out(self.curvy).to(self.device)
		# f=open('out.txt','w')
		# print('&&&&&&&&&&&&&&&&&&&&&&')
		# print(out_weight.size())
		# print(out_weight,file=f)
		# f.close()

		# print(self.w_in.size())
		# print(self.w_out.size())
		# print('&&&&&&&&&&&&&&&&&&&&&&')
		# self.w_in = self.w_in * out_weight
		# self.w_out = self.w_out * out_weight
		# print(self.w_in.size())
		# print(self.w_out.size())
		# print(self.w_loop.size())
		# print('&&&&&&&&&&&&&&&&&&&&&&')
		# exit()
		

		rel_embed = torch.cat([rel_embed, self.loop_rel], dim=0)
		num_edges = edge_index.size(1) // 2
		num_ent   = x.size(0)

		self.in_index, self.out_index = edge_index[:, :num_edges], edge_index[:, num_edges:]
		self.in_type,  self.out_type  = edge_type[:num_edges], 	 edge_type [num_edges:]

		self.loop_index  = torch.stack([torch.arange(num_ent), torch.arange(num_ent)]).to(self.device)
		self.loop_type   = torch.full((num_ent,), rel_embed.size(0)-1, dtype=torch.long).to(self.device)

		self.in_norm     = self.compute_norm(self.in_index,  num_ent)
		self.out_norm    = self.compute_norm(self.out_index, num_ent)
		
		in_res		= self.propagate('add', self.curv, self.in_index,   x=x, edge_type=self.in_type,   rel_embed=rel_embed, edge_norm=self.in_norm, 	mode='in')
		loop_res	= self.propagate('add', self.curv, self.loop_index, x=x, edge_type=self.loop_type, rel_embed=rel_embed, edge_norm=None, 		mode='loop')
		out_res		= self.propagate('add', self.curv, self.out_index,  x=x, edge_type=self.out_type,  rel_embed=rel_embed, edge_norm=self.out_norm,	mode='out')
		out		= self.drop(in_res)*(1/3) + self.drop(out_res)*(1/3) + loop_res*(1/3)


		if self.p.bias: out = out + self.bias
		out = self.bn(out)

		return self.act(out), torch.matmul(rel_embed, self.w_rel)[:-1]		# Ignoring the self loop inserted

	def rel_transform(self, ent_embed, rel_embed):
		if   self.p.opn == 'corr': 	trans_embed  = ccorr(ent_embed, rel_embed)
		elif self.p.opn == 'sub': 	trans_embed  = ent_embed - rel_embed
		elif self.p.opn == 'mult': 	trans_embed  = ent_embed * rel_embed
		else: raise NotImplementedError

		return trans_embed

	def message(self, x_j, curv, edge_type, rel_embed, edge_norm, mode, edge_index):

		weight 	= getattr(self, 'w_{}'.format(mode))
		rel_emb = torch.index_select(rel_embed, 0, edge_type)
		xj_rel  = self.rel_transform(x_j, rel_emb)
		out	= torch.mm(xj_rel, weight)
									
		# print(x_j.size()) torch.Size([86835, 100])
		# print(rel_emb.size()) torch.Size([86835, 100])
		# print(xj_rel.size()) torch.Size([86835, 100])
		# print(out.size()) torch.Size([86835, 200])
		# print("************************************************")
		# curvlist=[]
		# for n1,n2 in edge_index.t():
		# 	if n1 == n2:
		# 		curvlist.append(1)
		# 	elif (mode == 'in'):
		# 		curvlist.append(curv[n1.item()][n2.item()]['ricciCurvature'])
		# 	elif (mode == 'out'):
		# 		curvlist.append(curv[n2.item()][n1.item()]['ricciCurvature'])
		# 	else:
		# 		curvlist.append(1)
		if(mode == 'in'):
			wcurv = softmax(self.curv, self.in_index[0])
			# for i in range(out_channels):
			out = torch.multiply(out,curv)
		elif (mode == 'out'):
			wcurv = softmax(self.curv, self.out_index[0])
			# for i in range(out_channels):
			out = torch.multiply(out,curv)

		#print(sys._getframe(0).f_code.co_filename)
		#print(sys._getframe(1).f_code.co_filename)
		#print(sys._getframe(0).f_code.co_name) # 当前函数名
		#print(sys._getframe(1).f_code.co_name)  # 调用该函数的函数名字，如果没有被调用，则返回<module>
		#print(sys._getframe(0).f_lineno) #当前函数的行号
		#print(sys._getframe(1).f_lineno) # 调用该函数的行号

		return out if edge_norm is None else out * edge_norm.view(-1, 1)

	def update(self, aggr_out):
		return aggr_out

	def compute_norm(self, edge_index, num_ent):
		row, col	= edge_index
		edge_weight 	= torch.ones_like(row).float()
		deg		= scatter_add(edge_weight, row, dim=0, dim_size=num_ent)	# Summing number of weights of the edges
		deg_inv		= deg.pow(-0.5)							# D^{-0.5}
		deg_inv[deg_inv	== float('inf')] = 0
		norm		= deg_inv[row] * edge_weight * deg_inv[col]			# D^{-0.5}

		return norm

	def __repr__(self):
		return '{}({}, {}, num_rels={})'.format(
			self.__class__.__name__, self.in_channels, self.out_channels, self.num_rels)


def create_wmlp(widths,nfeato,lbias):
    mlp_modules=[]
    for k in range(len(widths)-1):
        mlp_modules.append(Linear(widths[k],widths[k+1],bias=False))
        mlp_modules.append(LeakyReLU(0.2,True))
    mlp_modules.append(Linear(widths[len(widths)-1],nfeato,bias=lbias))
    return seq(*mlp_modules)
