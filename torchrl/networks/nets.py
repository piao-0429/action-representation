import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchrl.networks.init as init


class ZeroNet(nn.Module):
    def forward(self, x):
        return torch.zeros(1)


class Net(nn.Module):
    def __init__(
            self, output_shape,
            base_type,
            append_hidden_shapes=[],
            append_hidden_init_func=init.basic_init,
            net_last_init_func=init.uniform_init,
            activation_func=F.relu,
            **kwargs):

        super().__init__()

        self.base = base_type(activation_func=activation_func, **kwargs)
        self.activation_func = activation_func
        append_input_shape = self.base.output_shape
        self.append_fcs = []
        for i, next_shape in enumerate(append_hidden_shapes):
            fc = nn.Linear(append_input_shape, next_shape)
            append_hidden_init_func(fc)
            self.append_fcs.append(fc)
            # set attr for pytorch to track parameters( device )
            self.__setattr__("append_fc{}".format(i), fc)
            append_input_shape = next_shape

        self.last = nn.Linear(append_input_shape, output_shape)
        net_last_init_func(self.last)

    def forward(self, x):
        out = self.base(x)

        for append_fc in self.append_fcs:
            out = append_fc(out)
            out = self.activation_func(out)

        out = self.last(out)
        return out

class FlattenNet(Net):
    def forward(self, input):
        out = torch.cat(input, dim = -1)
        return super().forward(out)


def null_activation(x):
    return x

class BootstrappedNet(Net):
    def __init__(self, output_shape, 
                 head_num = 4,
                 **kwargs ):
        self.head_num = head_num
        self.origin_output_shape = output_shape
        output_shape *= self.head_num
        super().__init__(output_shape = output_shape, **kwargs)

    def forward(self, x, idx):
        base_shape = x.shape[:-1]
        out = super().forward(x)
        out_shape = base_shape + torch.Size([self.origin_output_shape, self.head_num])
        view_idx_shape = base_shape + torch.Size([1, 1])
        expand_idx_shape = base_shape + torch.Size([self.origin_output_shape, 1])
        
        out = out.reshape(out_shape)
        
        idx = idx.view(view_idx_shape)
        idx = idx.expand(expand_idx_shape)

        out = out.gather(-1, idx).squeeze(-1)
        return out


class FlattenBootstrappedNet(BootstrappedNet):
    def forward(self, input, idx ):
        out = torch.cat( input, dim = -1 )
        return super().forward(out, idx)

class EmbeddingNet_v1(nn.Module):
    def __init__(
            self, output_shape,
            task_input_shape,
            representation_shape,
            embedding_shape,
            base_type,
            append_hidden_shapes_task=[],
            append_hidden_shapes_action=[],
            append_hidden_init_func=init.basic_init,
            net_last_init_func=init.uniform_init,
            activation_func=F.relu,
            cat_state_task=False,
            save_embed = False,
            **kwargs):

        super().__init__()
        self.save_flag = save_embed
        self.cat_state_task = cat_state_task
        self.base = base_type(activation_func=activation_func, **kwargs)
        self.activation_func = activation_func
        append_input_shape = self.base.output_shape
        self.representation = nn.Linear(append_input_shape, representation_shape)
        if cat_state_task:
            append_input_shape = task_input_shape + self.base.input_shape
        else:
            append_input_shape = task_input_shape
        self.append_fcs_task = []
        for i, next_shape in enumerate(append_hidden_shapes_task):
            fc = nn.Linear(append_input_shape, next_shape)
            append_hidden_init_func(fc)
            self.append_fcs_task.append(fc)
            # set attr for pytorch to track parameters( device )
            self.__setattr__("append_fc_task{}".format(i), fc)
            append_input_shape = next_shape
        self.embedding = nn.Linear(append_input_shape, embedding_shape)

        append_input_shape = self.base.output_shape
        self.representation = nn.Linear(append_input_shape, representation_shape)
        append_input_shape=representation_shape+embedding_shape
        self.append_fcs_action = []
        for i, next_shape in enumerate(append_hidden_shapes_action):
            fc = nn.Linear(append_input_shape, next_shape)
            append_hidden_init_func(fc)
            self.append_fcs_action.append(fc)
            # set attr for pytorch to track parameters( device )
            self.__setattr__("append_fc{}".format(i), fc)
            append_input_shape = next_shape
        self.last=nn.Linear(append_input_shape, output_shape)
        net_last_init_func(self.last)
    
    def forward(self, x, task_input):
        represent=self.representation(self.base(x))
        if self.cat_state_task:
            embed = torch.cat([x, task_input], dim=-1)
        else:
            embed = task_input
        for append_fc in self.append_fcs_task:
            embed = append_fc(embed)
            embed = self.activation_func(embed)
        embed=self.embedding(embed)

        out=torch.cat([represent,embed],dim=-1)
        for append_fc in self.append_fcs_action:
            out = append_fc(out)
            out = self.activation_func(out)
        out=self.last(out)
        if self.save_flag:
            return out, embed
        return out, None




class EmbeddingNet_v2(nn.Module):
    def __init__(
            self, output_shape,
            task_input_shape,
            representation_shape,
            embedding_shape,
            base_type,
            append_hidden_shapes_task=[],
            append_hidden_shapes_action=[],
            append_hidden_init_func=init.basic_init,
            net_last_init_func=init.uniform_init,
            activation_func=F.relu,
            cat_state_task=False,
            save_embed = False,
            **kwargs):

        super().__init__()
        self.save_flag = save_embed
        self.cat_state_task = cat_state_task
        self.base = base_type(activation_func=activation_func, **kwargs)
        self.activation_func = activation_func
        append_input_shape = self.base.output_shape
        self.representation = nn.Linear(append_input_shape, representation_shape)
        if cat_state_task:
            append_input_shape = task_input_shape + self.base.input_shape
        else:
            append_input_shape = task_input_shape
        self.append_fcs_task = []
        for i, next_shape in enumerate(append_hidden_shapes_task):
            fc = nn.Linear(append_input_shape, next_shape)
            append_hidden_init_func(fc)
            self.append_fcs_task.append(fc)
            # set attr for pytorch to track parameters( device )
            self.__setattr__("append_fc_task{}".format(i), fc)
            append_input_shape = next_shape
        self.embedding = nn.Linear(append_input_shape, embedding_shape)

        append_input_shape = self.base.output_shape
        self.representation = nn.Linear(append_input_shape, representation_shape)
        append_input_shape=representation_shape+embedding_shape
        self.append_fcs_action = []
        for i, next_shape in enumerate(append_hidden_shapes_action):
            fc = nn.Linear(append_input_shape, next_shape)
            append_hidden_init_func(fc)
            self.append_fcs_action.append(fc)
            # set attr for pytorch to track parameters( device )
            self.__setattr__("append_fc{}".format(i), fc)
            append_input_shape = next_shape
        self.last=nn.Linear(append_input_shape, output_shape)
        net_last_init_func(self.last)
    
    def forward(self, x, task_input):
        represent=self.representation(self.base(x))
        # if self.cat_state_task:
        #     embed = torch.cat([x, task_input], dim=-1)
        # else:
        #     embed = task_input
        # for append_fc in self.append_fcs_task:
        #     embed = append_fc(embed)
        #     embed = self.activation_func(embed)
        # embed=self.embedding(embed)
        embed = torch.Tensor([[-0.158629075,-0.05762929,0.16613258,-0.231542735,-0.265266689,0.247143658,1.46225075,0.23818919,-0.086032864,-0.6052495,	-0.055334119,-0.029710205,0.005168539,0.035747554,0.044847204,-0.028831988,-0.122371268,0.804044775,-1.01528197,0.077465673,-0.24746594,0.046888215,-0.244170335,-0.07828709,-0.65302369,-0.150723586,-0.16394408,-0.14125984,0.017955065,-0.51090087,0.2224715,-0.14848531]]) 
        embed = torch.Tensor([[-0.13289509	,0.12309803,	0.158302755,	-0.192204535	,-0.086660924	,0.71597398	,1.0904626	,-0.077608577	,0.102690511	,-0.00938375	,-0.019366114	,-0.39880861	,0.020456175	,-0.04960794	,0.103562543	,-0.208707098	,0.07022235,	0.274018385,	-0.68806612,	-0.02583901,	-0.874672735,	-0.24366121,	0.073069565	,-0.27975573,	-0.07978519	,-0.032591376,	-0.4486416	,-0.078896705,	0.285401115	,-0.95274325,	0.5663592,-0.1642595]])
        out=torch.cat([represent,embed],dim=-1)
        for append_fc in self.append_fcs_action:
            out = append_fc(out)
            out = self.activation_func(out)
        out=self.last(out)
        if self.save_flag:
            return out, embed
        return out, None

# class EmbeddingNet_v2(nn.Module):
#     def __init__(
#             self, output_shape,
#             representation_shape,
#             embedding_shape,
#             base_type,
#             append_hidden_shapes_action=[],
#             append_hidden_init_func=init.basic_init,
#             net_last_init_func=init.uniform_init,
#             activation_func=F.relu,
#             **kwargs):

#         super().__init__()
#         self.base = base_type(activation_func=activation_func, **kwargs)
#         self.activation_func = activation_func
#         append_input_shape = self.base.output_shape
#         self.representation = nn.Linear(append_input_shape, representation_shape)

#         append_input_shape=representation_shape+embedding_shape
#         self.append_fcs_action = []
#         for i, next_shape in enumerate(append_hidden_shapes_action):
#             fc = nn.Linear(append_input_shape, next_shape)
#             append_hidden_init_func(fc)
#             self.append_fcs_action.append(fc)
#             # set attr for pytorch to track parameters( device )
#             self.__setattr__("append_fc{}".format(i), fc)
#             append_input_shape = next_shape
#         self.last=nn.Linear(append_input_shape, output_shape)
#         net_last_init_func(self.last)
    
#     def forward(self, x, embedding_input):
#         represent=self.representation(self.base(x))
        

#         out=torch.cat([represent,embedding_input],dim=-1)
#         for append_fc in self.append_fcs_action:
#             out = append_fc(out)
#             out = self.activation_func(out)
#         out=self.last(out)
#         return out,