import torch
import torch.nn as nn

def profile_conv3d(mod, inp, outp): # model, input, output
    Bias_add = 1 if mod.bias is not None else 0
    ChIn = mod.in_channels // mod.groups
    Ops = ChIn * mod.kernel_size[0] * mod.kernel_size[1] * mod.kernel_size[2] + (ChIn * mod.kernel_size[0] * mod.kernel_size[1] * mod.kernel_size[2] -1) + Bias_add
    TotalOps = outp.numel() * Ops
    TotalSynps = (mod.out_channels // mod.groups) * ChIn * mod.kernel_size[0] * mod.kernel_size[1] * mod.kernel_size[2]
    #print("Conv3d ops {:.2e}".format(TotalOps))
    TotalNeurons = outp.numel() 
    mod.TotalOps += torch.Tensor([int(TotalOps)])
    mod.TotalSynps += torch.Tensor([int(TotalSynps)])
    mod.TotalNeurons += torch.Tensor([int(TotalNeurons)])

def profile_conv2d(mod, inp, outp):
    Bias_add = 1 if mod.bias is not None else 0
    ChIn = mod.in_channels // mod.groups
    Ops = ChIn * mod.kernel_size[0] * mod.kernel_size[1] + (ChIn * mod.kernel_size[0] * mod.kernel_size[1] -1) + Bias_add # number of operations: multi + addition + biasses_add
    TotalOps = outp.numel() * Ops
    TotalSynps = (mod.out_channels // mod.groups) * ChIn * mod.kernel_size[0] * mod.kernel_size[1]
    TotalNeurons = outp.numel()
    #print ("out size ", outp.size())
    #print("Conv2d neurons (K) ", TotalNeurons/1e3)
    mod.TotalOps += torch.Tensor([int(TotalOps)])
    mod.TotalSynps += torch.Tensor([int(TotalSynps)])
    mod.TotalNeurons += torch.Tensor([int(TotalNeurons)])


def profile_linear(mod, inp, outp):
    TotalOps = (mod.in_features + (mod.in_features -1)) * outp.numel() # (mult+addition)*num_out
    mod.TotalOps += torch.Tensor([int(TotalOps)])
    TotalSynps = outp.numel() * mod.in_features 
    TotalNeurons = outp.numel()
    mod.TotalSynps += torch.Tensor([int(TotalSynps)])
    mod.TotalNeurons += torch.Tensor([int(TotalNeurons)])
    #print("FC neurons (K) ", TotalNeurons/1e3)


def profile_BatchNorm3d(mod, inp, outp):
    TotalOps = inp[0].numel() + inp[0].numel() # there is one substraction and one division per input element 
    mod.TotalOps += torch.Tensor([int(TotalOps)])

def profile_BatchNorm2d(m, inp, outp):    
    TotalOps = inp[0].numel() + inp[0].numel() # there is one substraction and one division per input element 
    mod.TotalOps += torch.Tensor([int(TotalOps)])

def profile_relu(mod, inp, outp):
    TotalOps = inp[0].numel()
    mod.TotalOps += torch.Tensor([int(TotalOps)])

def profile_sigmoid(mod, inp, outp):
    ExpOps = inp[0].numel()
    AddOps = inp[0].numel()
    DivOps = inp[0].numel()
    Ops = ExpOps + AddOps + DivOps  
    mod.TotalOps += torch.Tensor([int(TotalOps)])

def profile_softmax(mod, inp, outp):
    BatchSize, NumFeatures = inp[0].size()
    TotalOps = BatchSize * (inp[0].size() + inp[0].size() + inp[0].size()) #there is one exp, one addition and one division per class(batch element) 
    mod.TotalOps += torch.Tensor([int(Ops)])

def profile_maxpool(mod, inp, outp):
    Ops = (torch.prod(torch.Tensor([mod.kernel_size])) - 1) * outp.numel()
    mod.TotalOps += torch.Tensor([int(Ops)])

def profile_avgpool(mod, inp, outp):
    Ops = (torch.prod(torch.Tensor([mod.kernel_size]) - 1) + 1) * outp.numel() #addition + div * output_elements 
    mod.TotalOps += torch.Tensor([int(Ops)])

def NN_Profiler(model, inp_size):

    model.eval()
    print ("model loaded")

    def Lookup(mod):
        if len(list(mod.children())) > 0: return
        mod.register_buffer('TotalOps', torch.zeros(1))
        mod.register_buffer('TotalParams', torch.zeros(1))
        mod.register_buffer('TotalNeurons', torch.zeros(1))
        mod.register_buffer('TotalSynps', torch.zeros(1))

        for param in mod.parameters():
            mod.TotalParams += torch.Tensor([param.numel()])

        if isinstance(mod, nn.Conv3d):
            mod.register_forward_hook(profile_conv3d)

        elif isinstance(mod, nn.Conv2d):
            mod.register_forward_hook(profile_conv2d)

        elif isinstance(mod, nn.Linear):
            mod.register_forward_hook(profile_linear)

        elif isinstance(mod, nn.Sigmoid):
            mod.register_forward_hook(profile_sigmoid)

        elif isinstance(mod, nn.ReLU):
            mod.register_forward_hook(profile_relu)

        elif isinstance(mod, nn.BatchNorm3d):
            mod.register_forward_hook(profile_BatchNorm3d)
	
        elif isinstance(mod, nn.BatchNorm2d):
            mod.register_forward_hook(profile_BatchNorm2d)

        elif isinstance(mod, (nn.MaxPool3d, nn.MaxPool2d)):
            mod.register_forward_hook(profile_maxpool)

        elif isinstance(mod, (nn.AvgPool3d, nn.AvgPool2d)):
            mod.register_forward_hook(profile_avgpool)

        else:
            print("Not yet implemented for the layer ", mod)

    model.apply(Lookup)
    model(torch.zeros(inp_size))

    Ops = 0
    Params = 0
    Neurons = 0 
    Synps = 0
    for modu in model.modules():
        if len(list(modu.children())) > 0: continue
        Ops += modu.TotalOps
        Params += modu.TotalParams
        Neurons += modu.TotalNeurons
        Synps += modu.TotalSynps
        #print(modu)

    return Ops, Params, Neurons, Synps
