import torch
import torch.nn as nn
import argparse
import torch.nn.functional as F
import pdb

class TriplePGM(nn.Module):
    def __init__(self,
                 entity_num: int,
                 relation_num: int,
                 type_num: int,
                 embedding_dim: int):
        super(TriplePGM, self).__init__()
        # \Phi
        self.entity_embedding = nn.Embedding(num_embeddings=entity_num,
                                             embedding_dim=embedding_dim)
        # \varphi
        self.soft_max_embedding = nn.Embedding(num_embeddings=entity_num,
                                               embedding_dim=embedding_dim)
        # \Psi or q_1(Tt | t), q_2(Th | h) distribution
        self.type_embedding = nn.Embedding(num_embeddings=type_num,
                                           embedding_dim=embedding_dim)

        # R_r
        # self.relation_matrix_list = []
        # for relation_id in range(relation_num):
        #     self.relation_matrix_list.append(nn.Linear(embedding_dim, 
        #                                                embedding_dim, bias=False))
        # self.relation_matrix_list = nn.ModuleList(self.relation_matrix_list)
        self.relation_embedding = nn.Embedding(num_embeddings=relation_num,
                                             embedding_dim=embedding_dim)
        self.init_emb()

    def init_emb(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, 
                h_r_t_id_tensor: torch.Tensor, 
                temp: float, 
                if_train: bool=True):
        # h_r_t_id_tensor [batch_size, 3]
        # h_idx, r_idx, t_idx = tuple(h_r_t_id_list)
        pdb.set_trace()
        h_idx = h_r_t_id_tensor[:, 0] # shape = [batch_size]
        r_idx = h_r_t_id_tensor[:, 1] # shape = [batch_size]
        t_idx = h_r_t_id_tensor[:, 2] # shape = [batch_size]
        phi_h = self.entity_embedding(h_idx) # [batch_size, embedding_dim]
        R_r = self.relation_embedding(r_idx) # [batch_size, embedding_dim]
        phi_t = self.entity_embedding(t_idx) # [batch_size, embedding_dim]
        # 用于计算头实体类型的KL散度
        q_1_Th_given_h_r_score = torch.matmul(phi_h, self.type_embedding.weight.T) #2 batch, type_num
        q_1_Th_given_h_r = F.softmax(q_1_Th_given_h_r_score, dim=-1) # 每个头实体种类类型的概率分布 [batch_size, type_num_of_head]
        prior_Th_given_h_r_score = torch.matmul(phi_h * R_r, self.type_embedding.weight.T) #3
        prior_Th_given_h_r = F.softmax(prior_Th_given_h_r_score, dim=-1)
        # 根据q_1的分布，实际抽样出一个对头实体类型的猜测。
        if if_train:
            Th_one_hot = F.gumbel_softmax(logits=q_1_Th_given_h_r_score, tau=temp, hard=True) # Th_one_hot [batch, type_num]
        else:
            tmp = q_1_Th_given_h_r_score.argmax(dim=-1).reshape(q_1_Th_given_h_r_score.shape[0], 1)
            Th_one_hot = torch.zeros(q.shape).to(self.device).scatter_(1, tmp, 1.)
        psi_Th = torch.mm(Th_one_hot, self.type_embedding.weight) # batch, embedding_dim
        # 根据猜测的头实体类型，求尾实体类型的KL散度
        q_2_Tt_given_t_score = torch.matmul(phi_t, self.type_embedding.weight.T) #4
        q_2_Tt_given_t = F.softmax(q_2_Tt_given_t_score, dim=-1) # [batch_size, type_num_of_head]
        prior_Tt_given_Th_r_score = torch.matmul(psi_Th + R_r, self.type_embedding.weight.T) #5
        prior_Tt_given_Th_r = F.softmax(prior_Tt_given_Th_r_score, dim=-1) # 对每一行进行softmax
        # 根据q_2的分布，实际抽样出一个对尾实体类型的猜测。
        if if_train:
            Tt_one_hot = F.gumbel_softmax(logits=q_2_Tt_given_t_score, tau=temp, hard=True)
        else:
            tmp = q_2_Tt_given_t_score.argmax(dim=-1).reshape(q_2_Tt_given_t_score.shape[0], 1)
            Tt_one_hot = torch.zeros(q.shape).to(self.device).scatter_(1, tmp, 1.)
        psi_Tt = torch.mm(Tt_one_hot, self.type_embedding.weight)
        # 求尾实体的重构误差
        # varphi_t = self.soft_max_embedding(t_idx) # [1, embedding_dim] 
        recon_t_given_Tt_score = torch.matmul(psi_Tt, self .soft_max_embedding.weight.T) #1
        # recon_t_given_Tt_score = psi_Tt #1 当采用负采样技术时，将psi_Tt同varphi_t一起返回。
        return recon_t_given_Tt_score, t_idx, q_1_Th_given_h_r, prior_Th_given_h_r, q_2_Tt_given_t, prior_Tt_given_Th_r
    
    
def loss_function(q_1_Th_given_h_r, prior_Th_given_h_r, 
                  q_2_Tt_given_t, prior_Tt_given_Th_r,
                  recon_t_given_Tt_score, t_idx):
    BCE = F.cross_entropy(recon_t_given_Tt_score, t_idx, reduction='sum') / t_idx.shape[0]

    log_q_1 = torch.log(q_1_Th_given_h_r + 1e-20)
    KLD_1 = torch.sum(q_1_Th_given_h_r*(log_q_1 - torch.log(prior_Th_given_h_r)),dim=-1).mean()
    
    log_q_2 = torch.log(q_2_Tt_given_t + 1e-20)
    KLD_2 = torch.sum(q_2_Tt_given_t*(log_q_2 - torch.log(prior_Tt_given_Th_r)),dim=-1).mean() # prior_Tt_given_Th_r 基本上全都为0，导致KLD_2为nan或者inf

    return BCE + KLD_1 + KLD_2
