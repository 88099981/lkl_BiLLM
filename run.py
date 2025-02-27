import time
import wandb
import torch
import torch.nn as nn
import argparse
import json

from typing import Any
from typing_extensions import Dict
from bigptq import BRAGPTQ
from binary import Binarization
from modelutils import find_layers

VISUALIZE = False

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='模型名称')
    parser.add_argument('--dataset', type=str, help='数据集名称')
    # 添加其他已有的参数...
    return parser.parse_args()

def get_model(model, model_dtype):
    import torch

    def skip(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    if "opt" in model:
        from transformers import OPTForCausalLM

        model = OPTForCausalLM.from_pretrained(model, torch_dtype=model_dtype)
        model.seqlen = model.config.max_position_embeddings
        print(model.dtype)
    elif "llama" in model:
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(model, torch_dtype=model_dtype) 
        model.seqlen = 2048
        print(model.dtype)
    return model


'''
The function is employed to calibrate and quantize models layer by layer.
'''
@torch.no_grad()
def quant_sequential(model, dataloader, dev, json_data=dict()):
    print("Starting ...")

    for name, module in model.named_modules():
        module.global_name = args.model + name

    use_cache = model.config.use_cache
    model.config.use_cache = False

    if "opt" in args.model:
        layers = model.model.decoder.layers
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(
            dev
        )
        if (
            hasattr(model.model.decoder, "project_out")
            and model.model.decoder.project_out
        ):
            model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
        if (
            hasattr(model.model.decoder, "project_in")
            and model.model.decoder.project_in
        ):
            model.model.decoder.project_in = model.model.decoder.project_in.to(dev)
    elif "llama" in args.model:
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)
        # model.model.rotary_emb = model.model.rotary_emb.to(dev) # 适配transformers 4.47.1
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {"i": 0, "attention_mask": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            cache['position_ids'] = kwargs['position_ids'] # 与gptq对齐，依据AutoGPTQ的issue，原因好像是库版本的问题
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    if "opt" in args.model:
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
        if (
            hasattr(model.model.decoder, "project_out")
            and model.model.decoder.project_out
        ):
            model.model.decoder.project_out = model.model.decoder.project_out.cpu()
        if (
            hasattr(model.model.decoder, "project_in")
            and model.model.decoder.project_in
        ):
            model.model.decoder.project_in = model.model.decoder.project_in.cpu()
    elif "llama" in args.model:
        model.model.embed_tokens = model.model.embed_tokens.cpu()
        model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]
    position_ids = cache['position_ids']

    #保存路径
    save_path = os.path.dirname(f'/home/liukunlong/lkl_BiLLM/output/{model.config.model_type}_{args.model_dtype}_{args.dataset}/')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_path_pth = os.path.join(save_path, 'pth')
    if not os.path.exists(save_path_pth):
        os.makedirs(save_path_pth)

    print("Ready.")

    
    
    for i in range(len(layers)):
        layer = layers[i].to(dev)

        subset = find_layers(layer)

        gptq = {}
        for name in subset:
            if (
                not (args.minlayer <= i < args.maxlayer and args.quant_only in name)
            ) == (not args.invert):
                continue
            braq_quantizer = Binarization(
                subset[name].weight,
                method=args.low_quant_method,
                groupsize=groupsize,
            )
            gptq[name] = BRAGPTQ(
                subset[name],
                braq_quantizer,
                salient_metric=args.salient_metric,
                disable_gptq=args.disable_gptq,
            )

        def add_batch(name):
            def tmp(_, inp, out):
                gptq[name].add_batch(inp[0].data, out.data)

            return tmp

        handles = []
        for name in gptq:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()

        for name in gptq:
            print(i, name)
            print("Quantizing ...")

            if f'{i}-{name}' not in json_data:
                json_data[f'{i}-{name}']={}
            info = gptq[name].fasterquant(
                percdamp=args.percdamp, 
                blocksize=args.blocksize,
                json_data=json_data[f'{i}-{name}'],
            )
            if VISUALIZE:
                json_data[f'{i}-{name}']={'dtype': f'{args.model_dtype}', 'salient_metric': f'{args.salient_metric}'}
                torch.save(gptq[name].layer.weight.data, f'{save_path_pth}/{i}_{name}.pth')
                torch.save(json_data[f'{i}-{name}'][f'Hinv'], f'{save_path_pth}/{i}_{name}_hinv.pth')
                del json_data[f'{i}-{name}'][f'Hinv']

            gptq[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        layers[i] = layer.cpu()
        del layer
        del gptq
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    if VISUALIZE:
        # 将json_data保存到文件
        current_time = time.strftime('%Y%m%d-%H%M%S')
        with open(f'{save_path}/{current_time}.json', 'w') as f:
            json.dump(json_data, f, indent=4)

    model.config.use_cache = use_cache


if __name__ == "__main__":   
    import argparse
    from datautils import *

    def list_of_ints(arg):
        return list(map(int, arg.split(',')))
    
    def list_of_floats(arg):
        return list(map(float, arg.split(',')))

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "model", type=str, help="model to load; for example `huggyllama/llama-7b`."
    )
    parser.add_argument(
        "dataset",
        type=str,
        choices=["wikitext2", "ptb", "c4"],
        help="Where to extract calibration data from.",
    )
    parser.add_argument(
        "low_quant_method",
        type=str,
        choices=["xnor", "sign", "no", "2bit", "4bit", "prune", "braq"],
        help="quantization method; `xnor` is the method using XNOR to adapt hardware calculation; `prune` is the method used in sparseGPTQ; braq is the method used in BiLLM",
    )
    parser.add_argument(
        "model_dtype",
        type=str,
        choices=["float32", "float16"],
        help="model data type; `fp32` is the float32 data type; `fp16` is the float16 data type.",
    )
    parser.add_argument("--load_quantized", action="store_true")
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed for sampling the calibration data."
    )
    parser.add_argument(
        "--nsamples", type=int, default=128, help="Number of calibration data samples."
    )
    parser.add_argument(
        "--percdamp",
        type=float,
        default=0.01,
        help="Percent of the average Hessian diagonal to use for dampening.",
    )
    parser.add_argument(
        "--blocksize",
        type=int,
        default=128,
        help="Blocksize to use for adaptive mask selection.",
    )
    parser.add_argument(
        "--salient_metric",
        type=str,
        default="magnitude",
        choices=["magnitude", "hessian", "lkl_hessian"],
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="set the device to use for quantization.",
    )
    parser.add_argument(
        "--disable_gptq",
        action="store_true",
        help="disable GPTQ for quantization.",
    )
    parser.add_argument(
        "--minlayer", type=int, default=-1, help="Quant all layers with id >= this."
    )
    parser.add_argument(
        "--maxlayer", type=int, default=1000, help="Quant all layers with id < this."
    )
    parser.add_argument(
        "--quant_only",
        type=str,
        default="",
        help="Quant only layers that contain this text.",
    )
    parser.add_argument("--invert", action="store_true", help="Invert subset.")
    parser.add_argument(
        "--save",
        action="store_true",
    )
    parser.add_argument(
        "--log_wandb", action="store_true", help="Whether to log to wandb."
    )
    parser.add_argument(
        "--visualize", action="store_true", help="Whether to visualize."
    )

    args = parser.parse_args()


    groupsize = args.blocksize

    device = args.device
    save_title = f"{os.path.basename(args.model.lower())}_{args.dataset}_{args.low_quant_method}_{groupsize}_{args.salient_metric}"
    save_file = "/home/liukunlong/lkl_BiLLM/output/models/" + save_title.replace("/", "_") + ".pt"


    if args.load_quantized:
        model = get_model("/home/liukunlong/lkl_BiLLM/output/models/llama-2-7b-hf_c4_braq_128_hessian.pt", 'auto').to(device)
        model.eval()
    else: # braq
        
        chosen_data = {}

        model_fp32 = get_model(args.model, getattr(torch, 'float32')).to(device)      
        model_fp32.eval()
        model=model_fp32
        
        tick = time.time()
        dataloader, testloader = get_loaders(
            args.dataset,
            nsamples=args.nsamples,
            seed=args.seed,
            model=args.model,
            seqlen=model.seqlen,
        )
        

        quant_sequential(model_fp32, dataloader, device, chosen_data)

        
        del model_fp32, model
        torch.cuda.empty_cache()

        model_fp16 = get_model(args.model, getattr(torch, 'float16')).to(device)
        model_fp16.eval()
        model=model_fp16
        quant_sequential(model_fp16, dataloader, device, chosen_data)
        print("quantization time:", time.time() - tick, "s")

    if args.save:
        save_path = os.path.dirname(save_file)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        model.save_pretrained(save_file)

    if args.log_wandb:
        # 初始化wandb
        wandb.init(
            project="lkl_BiLLM_Quantization",
            name=f"{args.model}-{time.strftime('%Y%m%d-%H%M%S')}",
            config={
                "model": args.model,
                "dataset": args.dataset,
                "dtype": args.model_dtype,
            }
        )
    if args.visualize:
        VISUALIZE = True

    for dataset in ["wikitext2", "ptb", "c4"]:
        dataloader, testloader = get_loaders(
            dataset, seed=args.seed, seqlen=model.seqlen, model=args.model
        )
        print(dataset)
        if "opt" in args.model:
            from eval_ppl_utils import opt_eval

            opt_eval(model, testloader, device, dataset, args.log_wandb)
        elif "llama" in args.model:
            from eval_ppl_utils import llama_eval
            print(model.dtype)
            llama_eval(model, testloader, device, dataset, args.log_wandb)