import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from collections import OrderedDict

def pruning_generate(model,px):
    parameters_to_prune =[]
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            parameters_to_prune.append((m,'weight'))
    parameters_to_prune = tuple(parameters_to_prune)
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=px,
    )

def see_remain_rate(model):
    sum_list = 0
    zero_sum = 0
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            sum_list = sum_list+float(m.weight.nelement())
            zero_sum = zero_sum+float(torch.sum(m.weight == 0))     
    

def rewind_weight(model_dict, target_model_dict_keys):

    new_dict = {}
    for key in target_model_dict_keys:
        if 'mask' not in key:
            if 'orig' in key:
                ori_key = key[:-5]
            else:
                ori_key = key 
            new_dict[key] = model_dict[ori_key]

    return new_dict


def pruning_generate_sn(model, px, initial_weight):
    total = 0
    total_nonzero = 0
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            total += m.weight_orig.data.numel()
            try:
                mask = m.weight_orig.data.abs().clone().gt(0).float().cuda()
            except:
                mask = m.weight.data.abs().clone().gt(0).float().cuda()
            total_nonzero += torch.sum(mask)
    conv_weights = torch.zeros(total)
    index = 0
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            try:
                size = m.weight_orig.data.numel()
                conv_weights[index:(index + size)] = m.weight_orig.data.view(-1).abs().clone()
            except:
                size = m.weight.data.numel()
                conv_weights[index:(index + size)] = m.weight.data.view(-1).abs().clone()
            index += size

    y, i = torch.sort(conv_weights)
    
    thre_index = total - total_nonzero + int(total_nonzero * px)
    thre = y[int(thre_index)]
    pruned = 0
    print('Pruning threshold: {}'.format(thre))
    zero_flag = False
    masks = OrderedDict()
    for k, m in enumerate(model.modules()):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            try:
                weight_copy = m.weight_orig.data.abs().clone()
            except:
                weight_copy = m.weight.data.abs().clone()
            mask = weight_copy.gt(thre).float()
            masks[k] = mask
            print("{}: {}, {}".format(k, m, mask.shape))
            pruned = pruned + mask.numel() - torch.sum(mask)
            m.weight_orig.data.mul_(mask)
            if int(torch.sum(mask)) == 0:
                zero_flag = True
            print('layer index: {:d} \t total params: {:d} \t remaining params: {:d}'.
                format(k, mask.numel(), int(torch.sum(mask))))
    print('Total conv params: {}, Pruned conv params: {}, Pruned ratio: {}'.format(total, pruned, pruned / total))
    # Load initial weights back
    model.load_state_dict(initial_weight)
    # Apply map
    for k, m in enumerate(model.modules()):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            try:
                m.weight_orig.data.mul_(masks[k])
            except:
                m.weight.data.mul_(masks[k])

    return model, masks

def see_remain_rate_orig(model):
    sum_list = 0.001
    zero_sum = 0
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            sum_list = sum_list+float(m.weight_orig.nelement())
            zero_sum = zero_sum+float(torch.sum(m.weight_orig == 0))     
    return 100*(1-zero_sum/sum_list)
