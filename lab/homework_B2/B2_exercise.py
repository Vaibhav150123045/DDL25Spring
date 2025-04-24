from simplellm.llama import LLamaFirstStage, LLamaLastStage, LLamaStage  # get our models
from simplellm.tokenizers import SPTokenizer  # get our tokenizer
from simplellm.dataloaders import TinyStories  # get our dataset
from simplellm.losses import causalLLMLoss  # our loss
from torch.optim import SGD, Adam
import torch.nn.functional as F
import torch
import torch.distributed as dist
import os
from sys import argv
rank = int(argv[1])
os.environ["MASTER_ADDR"] = "localhost"
world_size = 6
os.environ["MASTER_PORT"] = "29500"
dist.init_process_group("gloo", rank=rank, world_size=world_size)
torch.manual_seed(0)
dmodel = 288
num_heads = 6
n_layers = 12 // world_size
seq_l = 256
batch_size = 9


device = torch.device("cpu")

sizes = []
len_sizes = []


num_micro_batches = 3
micro_batch_size = batch_size // num_micro_batches  # Define micro-batch size

# Define process ranks for two groups
group1_ranks = [0, 2, 4]
group2_ranks = [1, 3, 5]

# Create groups
group1 = dist.new_group(group1_ranks)
group2 = dist.new_group(group2_ranks)


def assign_group(rank):
    if rank % 2 == 0:
        return group1
    else:
        return group2


# make the model
if rank == 0 or rank == 1:
    tokenizer = SPTokenizer()
    net = LLamaFirstStage(tokenizer.vocab_size, dmodel=dmodel, num_heads=num_heads,
                          device=device, n_layers=n_layers, ctx_size=seq_l)
    ds = TinyStories(tokenizer, batch_size=micro_batch_size,
                     seq_l=seq_l)  # no skip
    iter_ds = iter(ds)
elif rank == 2 or rank == 3:
    net = LLamaStage(dmodel=dmodel, num_heads=num_heads,
                     device=device, n_layers=n_layers, ctx_size=seq_l)
elif rank == 4 or rank == 5:
    tokenizer = SPTokenizer()
    net = LLamaLastStage(tokenizer.vocab_size, dmodel=dmodel, num_heads=num_heads,
                         device=device, n_layers=n_layers, ctx_size=seq_l)
    ds = TinyStories(tokenizer, batch_size=micro_batch_size,
                     seq_l=seq_l)  # no skip
    iter_ds = iter(ds)


optim = Adam(net.parameters(), lr=8e-4)

for param in net.parameters():
    sizes.append(param.shape)
    len_sizes.append(len(param.view(-1)))

for itr in range(900):
    optim.zero_grad()
    # Process micro-batches sequentially
    for batch_id in range(num_micro_batches):
        # FORWARD PASS:
        if rank == 0 or rank == 1:
            out = next(iter_ds)
            out = out.to(device)
            out = net.embed(out)

            send_req = dist.isend(out.to("cpu"), rank+2,
                                  tag=batch_id, group=assign_group(rank))
            send_req.wait()  # Ensure data is sent before proceeding

        elif rank == 2 or rank == 3:
            inp_batch = torch.empty((micro_batch_size, seq_l, dmodel))
            recv_req = dist.irecv(
                inp_batch, rank-2, tag=batch_id, group=assign_group(rank))
            recv_req.wait()  # Ensure data is received before processing
            with torch.no_grad():
                inp_batch = inp_batch.to(device)
                inp_batch.requires_grad_()
                inp_batch.retain_grad()

            out = net(inp_batch)
            send_req = dist.isend(out.to("cpu"), rank+2,
                                  tag=batch_id, group=assign_group(rank))
            send_req.wait()

        elif rank == 4 or rank == 5:
            target = next(iter_ds)
            inp_batch = torch.empty((micro_batch_size, seq_l, dmodel))
            recv_req = dist.irecv(
                inp_batch, rank-2, tag=batch_id, group=assign_group(rank))
            recv_req.wait()
            with torch.no_grad():
                inp_batch = inp_batch.to(device)
                inp_batch.requires_grad_()
                inp_batch.retain_grad()

            logits = net(inp_batch)
            loss = causalLLMLoss(logits, target, tokenizer.vocab_size)
            print(loss.item())
            loss.backward()

        # BACKWARD PASS:
        if rank == 4 or rank == 5:
            send_req = dist.isend(inp_batch.grad.to(
                "cpu"), rank-2, tag=batch_id, group=assign_group(rank))
            send_req.wait()
        elif rank == 2 or rank == 3:
            inp_grad = torch.empty((micro_batch_size, seq_l, dmodel))
            recv_req = dist.irecv(
                inp_grad, rank+2, tag=batch_id, group=assign_group(rank))
            recv_req.wait()
            out.backward(inp_grad.to(device))
            send_req = dist.isend(inp_batch.grad.to(
                "cpu"), rank-2, tag=batch_id, group=assign_group(rank))
            send_req.wait()
        elif rank == 0 or rank == 1:
            inp_grad = torch.empty((micro_batch_size, seq_l, dmodel))
            recv_req = dist.irecv(
                inp_grad, rank+2, tag=batch_id, group=assign_group(rank))
            recv_req.wait()
            out.backward(inp_grad.to(device))
        optim.step()
        torch.cuda.empty_cache()
