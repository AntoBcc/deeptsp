import numpy as np
from scipy.spatial.distance import pdist, squareform
import torch


def cost_matrix(graph):
    '''
    given a graph (in concorde format), compute the cost matrix
    using euclidean distances between nodes as costs

    graph:: list of nodes coordinates (x,y)
    '''

    num_nodes = len(graph)//2
    nodes_coord = []

    for i in range(0, 2 * num_nodes, 2):
        nodes_coord.append([float(graph[i]), float(graph[i + 1])])

    w = torch.tensor(squareform(pdist(nodes_coord, metric='euclidean')))

    return w


def initialize_msg(w,rand=False):
    
    '''
    Initialize matrix of messages associated to a graph, 
    as all zeros with -Inf on the diagonal

    w:: cost matrix of the graph
    '''

    assert w.shape[0] == w.shape[1]

    if rand:
        h = torch.rand_like(w) 
    else:
        h = torch.zeros_like(w)

    h.fill_diagonal_(-np.inf)
    
    return h

def oneBPiter(h, w):

    '''
    Perform one iteration of Belief Propagation

    h::matrix of messages
    w::cost matrix
    '''

    hnew = h.clone()
    n = h.shape[0]

    u = w - h
    k_min, ind = u.topk(3, dim=0, largest=False)
    hnew[:,:] = k_min[1] 
    hnew[ind[1],torch.arange(n)] = k_min[2]
    hnew[ind[0],torch.arange(n)] = k_min[2]
    hnew.fill_diagonal_(-np.inf)

    return hnew.t() 


def compute_matching(h,w,b,alt=False):
    
    '''
    return the configuration associated with current message updates

    if alt==True, use alternative function to obtain matching. 
    '''

    if alt:
        return __compute_matching_alt(h,w,b)
    else:
        return __compute_matching(h,w)


def __compute_matching(h,w):

    '''
    derivation loosely based on: Mézard, M. and Montanari, A. (2009). 'Information, Physics, and Computation.'

    h:: matrix of messages
    w:: cost matrix
    '''

    assert h.shape == w.shape
    n = torch.ones_like(w)
    hT = torch.transpose(h,0,1)
    n = n.masked_fill(h+hT-w<=0,0)
    return n 


def __compute_matching_alt(h,w,b=2):

    '''
    as in: Bayati et al. (2011), 'Belief Propagation for Weighted b-Matchings 
    on Arbitrary Graphs and its Relation to Linear Programs with Integer Solutions',
    SIAM J. Discrete Math., 25(2), 989–1011

    h:: matrix of messages
    w:: cost matrix
    b:: number of messages to consider, ordered from the lowest
    '''

    u = w - h
    _, ind = u.topk(b, dim=0, largest=False)
    match = ind.t()
    nnodes = len(h)
    n = torch.zeros_like(h, dtype=int)

    for i in range(b):
        n[torch.arange(nnodes),match[:,i]] = 1
        n[match[:,i],torch.arange(nnodes)] = 1

    return n


def count_violations(n):
    
    '''
    computes number of violated constraints (total is 2*numnodes):
    the sum of each row and each column should be b, for the b-matching to be perfect. 

    n::configuration matrix 
    '''

    return np.int(sum(torch.sum(n,0)!=2) + sum(torch.sum(n,1)!=2))


def twof_BP(graph,max_iter=1000,thresh=10,d=0,b=2,verbose=False,rand_init=False,random_seed=0,alt=False):

    '''
    Perform Belief Propagation for Minimum Cost 2-Matching

    graph:: list of nodes coordinates (x,y)
    max_iter:: max number of iterations before returning, whether converged or not
    thresh:: no of iterations to consider to verify convergence
    d:: damping coefficient
    b:: number of smallest messages to consider to compute the matching
    verbose:: print out information on convergence and number of violations
    rand_init:: whether to initialize messages randomly in [0,1) (else, all 0)
    random_seed:: torch seed to use, relevant if rand_init=True
    '''

    torch.manual_seed(random_seed)

    w = cost_matrix(graph)
    num_nodes = w.shape[0]
    h = initialize_msg(w,rand=rand_init)
    assert  w.shape == h.shape  
    conv_iters = 0
    n_prev = compute_matching(h,w,b,alt)
    converged = False

    iters = 0

    for it in range(max_iter):

        iters += 1

        old_h = h
        new_h = oneBPiter(h,w)
        h = d * old_h + (1-d) * new_h
        n = compute_matching(h,w,b,alt)
        
        if torch.equal(n_prev,n):
            conv_iters +=1  
        else:
            conv_iters = 0

        n_prev = n
        
        if conv_iters == thresh:
            converged = True
            break

    violations = count_violations(n)
    cost = int(torch.sum(torch.masked_select(w,n==1)))

    if verbose:
        if converged: 
            print(f'Instance has converged in {iters}/{max_iter} iterations') 
        else: 
            print(f'Instance has converged in {iters}/{max_iter} iterations')

        print(f'Number of contraints violated: {violations} - {(violations/n.shape[0]):.1%}')
        print(f'Total cost of final configuration: {cost}')
        print('-'*20)

    return n, violations, converged, cost



# %%
