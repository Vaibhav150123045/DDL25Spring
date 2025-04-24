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
world_size = 3
os.environ["MASTER_PORT"] = "29500"
dist.init_process_group("gloo", rank=rank, world_size=world_size)
torch.manual_seed(0)
dmodel = 288
num_heads = 6
n_layers = 6 // world_size
seq_l = 256
batch_size = 18  # increased batch_size because we will make smaller micro batches out of it

# assigning the device value
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# Defining number of microbatches and size of each microbatch
num_micro_batches = 6
micro_batch_size = batch_size // num_micro_batches  # Define micro-batch size


# make the tokenizer

# make the model
if rank == 0:
    tokenizer = SPTokenizer()
    net = LLamaFirstStage(tokenizer.vocab_size, dmodel=dmodel, num_heads=num_heads,
                          device=device, n_layers=n_layers, ctx_size=seq_l)
    # Provided batch_size = micro_batch_size because we will be dealing with one microbatch at a time
    ds = TinyStories(tokenizer, batch_size=micro_batch_size,
                     seq_l=seq_l)  # no skip.
    iter_ds = iter(ds)
elif rank == 1:
    net = LLamaStage(dmodel=dmodel, num_heads=num_heads,
                     device=device, n_layers=n_layers, ctx_size=seq_l)
elif rank == 2:
    tokenizer = SPTokenizer()
    net = LLamaLastStage(tokenizer.vocab_size, dmodel=dmodel, num_heads=num_heads,
                         device=device, n_layers=n_layers, ctx_size=seq_l)
    # Provided batch_size = micro_batch_size because we will be dealing with one microbatch at a time
    ds = TinyStories(tokenizer, batch_size=micro_batch_size,
                     seq_l=seq_l)  # no skip.
    iter_ds = iter(ds)


optim = Adam(net.parameters(), lr=8e-4)

for itr in range(900):
    optim.zero_grad()
    # Process micro-batches sequentially. Each batch_id is index for a micro batch
    for batch_id in range(num_micro_batches):
        # FORWARD PASS:
        if rank == 0:
            out = next(iter_ds)
            out = out.to(device)
            out = net.embed(out)

            # using isend for asynchronous communication, also tag is populated to understand data is for which micro batch
            send_req = dist.isend(out.to("cpu"), 1, tag=batch_id)
            send_req.wait()  # Ensure data is sent before proceeding

        elif rank == 1:

            inp_batch = torch.empty((micro_batch_size, seq_l, dmodel))
            # using irecv for asynchronous communication, also tag is populated to understand data is for which micro batch
            recv_req = dist.irecv(inp_batch, 0, tag=batch_id)
            recv_req.wait()  # Ensure data is received before processing
            with torch.no_grad():
                inp_batch = inp_batch.to(device)
                inp_batch.requires_grad_()
                inp_batch.retain_grad()

            out = net(inp_batch)
            # using isend for asynchronous communication, also tag is populated to understand data is for which micro batch
            send_req = dist.isend(out.to("cpu"), 2, tag=batch_id)
            send_req.wait()  # Ensure data is sent before proceeding

        elif rank == 2:
            target = next(iter_ds)
            inp_batch = torch.empty((micro_batch_size, seq_l, dmodel))
            recv_req = dist.irecv(inp_batch, 1, tag=batch_id)
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
        if rank == 2:
            send_req = dist.isend(inp_batch.grad.to("cpu"), 1, tag=batch_id)
            send_req.wait()
        elif rank == 1:
            inp_grad = torch.empty((micro_batch_size, seq_l, dmodel))
            recv_req = dist.irecv(inp_grad, 2, tag=batch_id)
            recv_req.wait()
            out.backward(inp_grad.to(device))
            send_req = dist.isend(inp_batch.grad.to("cpu"), 0, tag=batch_id)
            send_req.wait()
        elif rank == 0:
            inp_grad = torch.empty((micro_batch_size, seq_l, dmodel))
            recv_req = dist.irecv(inp_grad, 1, tag=batch_id)
            recv_req.wait()
            out.backward(inp_grad.to(device))

        optim.step()
        torch.cuda.empty_cache()
