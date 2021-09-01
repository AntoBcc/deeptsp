import argparse
import numpy as np
import os
import sys
import torch 
import BP

description='run the BP algorithm for minimum cost 2-matching on a folder of concorde graph instances'

parser = argparse.ArgumentParser(description=description)
parser.add_argument('path', help='path to folder containing datasets')
parser.add_argument('-max_iter',type=int,default=1000,help='Max number of BP iterations')
parser.add_argument('-th', default=10,type=int,help='Number of subsequent iters to check for convergence')
parser.add_argument('-d',type=float,default=0,help='Dumping coefficient (default = no dumping)')
parser.add_argument('-b',default=2,type=int,help='number of messages to consider to produce matching')
parser.add_argument('-v',default=1,choices=[0,1,2],help='output verbosity. 0 = no info, 1 = info on progress, 2 = individual graph info')
parser.add_argument('-rand',default=0,help='random initialization of messages (default=False)')
parser.add_argument('-seed',default=0,type=int,help='seed for random initialization')
parser.add_argument('-samples',default=0,type=int,help='number of instances to consider (default = entire file')
parser.add_argument('-info',default=1,help='store cost, convergence and violation diagnostics')
parser.add_argument('-alt',default=0, help='use alternative method of producing matching at time t')

args = parser.parse_args()


f = args.path
max_iter = args.max_iter
thresh = args.th
d = args.d
b = args.b
verb = args.v
rand_init = args.rand
seed = args.seed
n_samples = args.samples
info = args.info
alt = args.alt

if not os.path.isdir(f):
    raise Exception(f'{f} is not a folder in the current directory')


verbose = (verb == 2)
out = 'BPmatch'
if d > 0:
    out += '_damp'
if alt:
    out += '_bayati'
if b > 2:
    out += f'_{b}' 
if not os.path.exists(out):
    os.makedirs(out)

#NEXT LINE TO BE FIXED

if not os.listdir(f):
    raise Exception(f'{f} is an empty folder')


for file in os.listdir(f):
    print(file)
    data = open(os.path.join(f,file), "r").readlines()
    if not n_samples: 
        n_samples = len(data)
    graphs = [data[i][:data[i].index('output')].split() for i in range(n_samples)]
    diagnostics = [['cost_of_matching','n_violations','converged']]
    matchings = []
    file_name = file.split('/')[-1][:-4] 

    for i in range(n_samples):
        n,v,conv,cost = BP.twof_BP(graphs[i],max_iter=max_iter,thresh=thresh,
        d=d,b=b,verbose=verbose,rand_init=rand_init,random_seed=seed,alt=alt)
        matchings.append(n)
        if info: 
            diagnostics.append([cost,v,conv])
        if verb == 1:
            print(f'Done {i+1}/{n_samples}')

    n_samples = 0
        
    #store matching configurations (as pytorch tensors)
    path = os.path.join(out,file_name+'_match.pt')
    path = os.path.join(out,file_name+'_match.pt')


    torch.save(matchings,os.path.join(out,file_name+'_match.pt'))

    #store diagnostics in text file 
    with open(os.path.join(out,file_name+'_match.txt'), "w") as outfile:
        outfile.write(f'max_iter={max_iter}, thresh={thresh}, d={d}, b={b}, rand_init={rand_init}, seed={seed}\n')
        outfile.write('\n'.join(' '.join(map(str, row)) for row in diagnostics))


