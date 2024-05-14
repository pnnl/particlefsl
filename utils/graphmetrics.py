"""
This material was prepared as an account of work sponsored by an agency of the
United States Government.  Neither the United States Government nor the United
States Department of Energy, nor Battelle, nor any of their employees, nor any
jurisdiction or organization that has cooperated in the development of these
materials, makes any warranty, express or implied, or assumes any legal
liability or responsibility for the accuracy, completeness, or usefulness or
any information, apparatus, product, software, or process disclosed, or
represents that its use would not infringe privately owned rights.

Reference herein to any specific commercial product, process, or service by
trade name, trademark, manufacturer, or otherwise does not necessarily
constitute or imply its endorsement, recommendation, or favoring by the United
States Government or any agency thereof, or Battelle Memorial Institute. The
views and opinions of authors expressed herein do not necessarily state or
reflect those of the United States Government or any agency thereof.

                 PACIFIC NORTHWEST NATIONAL LABORATORY
                              operated by
                                BATTELLE
                                for the
                   UNITED STATES DEPARTMENT OF ENERGY
                    under Contract DE-AC05-76RL01830
"""

import numpy as np
import networkx as nx
import itertools
from math import comb
import json
import pandas as pd
import os.path as op
import scipy.stats as stats

try:
    import igraph as ig
    has_igraph=True
except:
    has_igraph=False

def merge(a, b):
    """Merge nested dictionaries.

    Args:
      a: Dictionary 1.
      b: Dictionary 2.

    Returns:
      Merged dictionary.
    """
    for key in b:
        if key in a:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                merge(a[key], b[key])
            elif a[key] == b[key]:
                pass 
            else:
                raise Exception(f'conflict at n_ints {key}: multiple different values found')
        else:
            a[key] = b[key]
    return a

set_map={'Chevy Caprise 422PC2 Front Driver':'Brake Dust',
         'Chevy Caprise 422PC2 Front Passenger':'Brake Dust',
         'Chevy Caprise 422PC2 Rear Driver':'Brake Dust',
         'Chevy Caprise 422PC2 Rear Passenger':'Brake Dust',
         'Debris from sparklers':'Firework Residue',
         'Ford Explorer A213 Front Driver':'Brake Dust',
         'Ford Explorer A213 Front Passenger':'Brake Dust',
         'Ford Explorer A213 Rear Driver':'Brake Dust',
         'Ford Explorer A213 Rear Passenger':'Brake Dust',
         'Ford Explorer B297 Front Driver':'Brake Dust',
         'Ford Explorer B297 Front Passenger':'Brake Dust',
         'Ford Explorer B297 Rear Driver':'Brake Dust',
         'Ford Explorer B297 Rear Passenger':'Brake Dust',
         'Roman Candles-Debris from JWC':'Firework Residue',
         'Roman Candles-Post-handling, pre-ignition':'Firework Residue',
         'Roman Candles-Post cleanup':'Firework Residue',
         'Shooter #1-Zero time':'Firearm Discharge',
         'Shooter #2-Zero time':'Firearm Discharge',
         'Shooter #3-Zero time L':'Firearm Discharge',
         'Shooter #3-Zero time R':'Firearm Discharge',
         'Shooter #4-Zero time L':'Firearm Discharge',
         'Shooter #4-Zero time R':'Firearm Discharge',
         'Shooter #5-Zero time L':'Firearm Discharge',
         'Shooter #5-Zero time R':'Firearm Discharge',
         'Sparklers during burn':'Firework Residue',
         'Sparklers post handling post burn':'Firework Residue',
         'Spinners-Debris from spinner':'Firework Residue',
         'Spinners-Post-cleanup':'Firework Residue',
         'Spinners-Post-handling, pre-ignition':'Firework Residue',
         'Spinners-Post-ignition':'Firework Residue'}

class simG:
    def __init__(self, 
                 data_repo='./data/processed'):
        """Class for creating a similarity graph and computing graph metrics.
        
        Args:
          data_repo: Top level directory containing processed data.
  
        """
        self.datarepo=data_repo
            
    def make(self, scorepath: str):
        """Create graph for first time without metrics"""
        self.scorepath=scorepath
        self.modelpath=scorepath.split('/inference/')[0]
        self.sample=scorepath.split('/inference/')[-1].split('/')[0]
        self.rawdata=op.join(self.datarepo,self.sample,self.sample+'.csv')
        try:
            self.set=set_map[self.sample]
        except:
            self.set='N/A'
            
        # make graph
        self.scores = pd.read_csv(self.scorepath)
        self.scores['weight']=self.scores['similarity']
        self.scores['x_l']=self.scores['x_l'].astype(str)
        self.scores['x_r']=self.scores['x_r'].astype(str)
        self.G=nx.from_pandas_edgelist(self.scores, 'x_l', 'x_r', ['weight', 'std'])
        
        # check that graph is a complete graph
        self._complete_check()
        
        self.data={'graph':{'nodes':self.G.number_of_nodes(), 
                            'edges':self.G.number_of_edges(), 
                            'complete': self.complete,
                            'representation':nx.to_dict_of_dicts(self.G)}, 
                   'data':{'sample': self.sample, 
                           'set': self.set,
                           'rawdata': self.rawdata},
                   'model':{'modelpath':self.modelpath,
                            'scorepath': self.scorepath}
                  }

    def _complete_check(self):
        """Check that graph is a complete graph"""
        n = self.G.number_of_nodes()
        self.complete = int((n*(n-1))/2) == self.G.number_of_edges()
        if not self.complete:
            print(f"Graph from {self.scorepath} is not a complete graph")
                
    def load(self, savepath: str):
        """Load graph and data"""
        with open(savepath, 'r') as fp:
            self.data = json.load(fp)
            
        # make graph
        self.G=nx.from_dict_of_dicts(self.data['graph']['representation'])
        
        # check that graph is a complete graph
        self._complete_check()
            
    def write(self, savepath: str):
        """Save graph data"""
        with open(savepath, 'w') as fp:
            json.dump(self.data, fp, sort_keys=False, indent=4)
            
    def add_to_data(self, dict_to_add, key_to_add):
        """Add to data dict"""
        self.data[key_to_add]=dict_to_add[key_to_add]

def thresh(G, t, isolates=False):
    """Compute subgraph of given edge-weighted graph restricted to edges of weight>t.
    Notes: isolated vertices may impact certain graph metrics, so the choice of whether or not
    to keep them in the thresholded graph may be impactful depending on the metric
    
    Args:
      G: NetworkX graph with edge weights between 0 and 1 under 'weight' parameter.
      t (float): Thresholding value
      isolates (bool): Whether to retain any isolate nodes that may result from restricting 
        the graph to edges of weight > t.
         
    Returns:
      NetworkX graph    
    """
    
    if isolates:
        nodes=list(G.nodes())
        H=G.edge_subgraph([e for e in G.edges() if G.edges[e]['weight']>t]).copy() #copy since frozen subgraph
        H.add_nodes_from(nodes) # edge induced subgraphs throw away isolated nodes, add them back
        return H
    else:
        return G.edge_subgraph([e for e in G.edges() if G.edges[e]['weight']>t])

### Fast diversity calculation ###

def diversity(G):
    m=G.number_of_edges()
    n=G.number_of_nodes()
    if n<=3:
        return 0

    density=m/(n*(n-1)/2)
    if density>0.248: #optimal value guess based on experimentation
        return _diversity_ESCAPE(G)
    else:
        return _diversity_sparse(G)

class graph(object):
    def __init__(self):
        #### Initializing empty graph ####
        self.adj_list = dict()   # Initial adjacency list is empty dictionary 
        self.vertices = set()    # Vertices are stored in a set   
        self.degrees = dict()    # Degrees stored as dictionary
        self.colors = dict()     # Colors assigned to each node in the graph
        
    def isEdge(self,node1,node2):
        if node1 in self.vertices:               # Check if node1 is vertex
            if node2 in self.adj_list[node1]:    # Then check if node2 is neighbor of node1
                return 1                         # Edge is present!

        if node2 in self.vertices:               # Check if node2 is vertex
            if node1 in self.adj_list[node2]:    # Then check if node1 is neighbor of node2 
                return 1                         # Edge is present!

        return 0                # Edge not present!

class DAG(graph):

    def __init__(self):
        super(DAG,self).__init__()
        DAG.top_order = []
        DAG.top_order_inv = dict()
        DAG.in_list = dict()            # Optional in-neighbor list. adj_list only maintains out neighbors
        DAG.indegrees = dict()          # Optional indegrees
        
def Orient(G,ordering):
    output = DAG()          # Creating empty output graph
    counter = 1
    for node in ordering:   # Loop over nodes
        output.vertices.add(node)   # First add node to vertices in output
        output.top_order_inv[node] = counter    # Setting inverse of topological ordering
        counter += 1
        output.adj_list[node] = set()  # Set up empty adjacency and in lists
        output.in_list[node] = set()
        output.degrees[node] = 0
        output.indegrees[node] = 0
        for nbr in G[node]: # For every neighbor nbr of node
            if nbr in output.vertices: # Determine which is higher in order. If nbr already in output.vertices, then nbr is lower. 
                output.in_list[node].add(nbr)  # If nbr is lower, then nbr is in-neighbor.
                output.indegrees[node] += 1       # Update degree of nbr accordingly. 
            else:
                output.adj_list[node].add(nbr) # If nbr is higher, nbr is out neighbor.
                output.degrees[node] += 1         # Update degree of nbr accordingly.          

    output.top_order = ordering         # Topological ordering is as given by input
    return output 

def DegenOrdering(G):
    n = G.number_of_nodes()
    touched = {}                 # Map of touched vertices
    cur_degs = {}                # Maintains degrees as vertices are processed
    top_order = []               # Topological ordering

    deg_list = [set() for _ in range(n)]    # Initialize list, where ith entry is set of deg i vertices
    min_deg = n       # variable for min degree of graph


    for node in G.nodes():    # Loop over nodes
        deg = G.degree(node)      # Get degree of node
        touched[node] = 0          # Node not yet touched
        cur_degs[node] = deg       # cur_degs of node just degree
        deg_list[deg].add(node)    # Update deg_list with node
        if deg < min_deg:          # Update min_deg
            min_deg = deg

    # At this stage, deg_list[d] is the list of vertices of degree d

    for i in range(n):        # The main loop, just going n times

        # We first need the vertex of minimum degree. Due to the looping and deletion of vertex, we may have exhaused
        # all vertices of minimum degree. We need to update the minimum degree

        while len(deg_list[min_deg]) == 0:  # update min_deg to reach non-empty set
            min_deg = min_deg+1

        source = deg_list[min_deg].pop()    # get vertex called "source" with minimum degree 
        top_order.append(source)     # append to this to topological ordering
        touched[source] = 1                 # source has been touched

        # We got the vertex of the ordering! All we need to do now is "delete" vertex from the graph,
        # and update deg_list appropriately.

        for node in G[source]: # loop over nbrs of source, each nbr called "node"

            if touched[node] == 1:         # if node has been touched, do nothing
                continue 

            # We update deg_list
            deg = cur_degs[node]           # degree of node
            deg_list[deg].remove(node)      # move node in deg_list, decreasing its degree by 1
            deg_list[deg-1].add(node)
            if deg-1 < min_deg:             # update min_deg in case node has lower degree
                min_deg = deg-1
            cur_degs[node] = deg-1          # decrement cur_deg because it has another touched neighbor


    return top_order

def triangle_info(DAG):
    tri_vertex = {}         # Output structures
    tri_edge = {}

    for node in DAG.vertices:
        tri_vertex[node] = 0.0   # Initialize for each vertex
        for nbr in DAG.adj_list[node]:
            tri_edge[(node,nbr)] = 0.0    # Initialize for each edge, note each edge is tuple with lower vertex first
            tri_edge[(nbr,node)] = 0.0    # Initialize for each edge, note each edge is tuple with lower vertex first

    for node1 in DAG.vertices:     # Loop over all nodes
        for (node2, node3) in itertools.combinations(DAG.adj_list[node1],2):    #Loop over all pairs of neighbors of node1
            if DAG.isEdge(node2,node3):    # If (node2, node3) form an edge
                tri_vertex[node1] += 1       # Increment all triangle counts
                tri_vertex[node2] += 1
                tri_vertex[node3] += 1
                tri_edge[(node1,node2)] += 1
                tri_edge[(node1,node3)] += 1
                tri_edge[(node2,node3)] += 1
                tri_edge[(node2,node1)] += 1
                tri_edge[(node3,node1)] += 1
                tri_edge[(node3,node2)] += 1
    return [tri_vertex, tri_edge]
    
def _diversity_dense(G):
    count=0
    m=G.number_of_edges()
    n=G.number_of_nodes()
    for x,y in itertools.combinations(G.edges(),2): #experiment with two for loops
        a,b=x
        c,d=y
        
        if G.has_edge(a,c) or G.has_edge(a,d) or G.has_edge(b,c) or G.has_edge(b,d):
            continue
        if a==c or a==d or b==c or b==d: # are the edges disjoint
            continue
            
        count+=1
    return np.sqrt(count/(comb(int(np.ceil(n/2)),2)*comb(int(np.floor(n/2)),2)))

def _diversity_sparse(g):
    G=nx.convert_node_labels_to_integers(g)
    m=G.number_of_edges()
    n=G.number_of_nodes()
    total=0
    for e in G.edges():
        a,b=e
        bad=set()
        for endpoint in [a,b]:
            if len(bad)==m:
                break
            for x in G[endpoint]: #first number of tuple, multiply by n, add it to second number
                if x<endpoint: 
                    bad.add(x*n+endpoint)
                else:
                    bad.add(endpoint*n+x)
                for y in G[x]:
                    if y<x:
                        bad.add(y*n+x)
                    else:
                        bad.add(x*n+y)
        total+=len(set(bad))
    return np.sqrt(((m**2-total)/2)/(comb(int(np.ceil(n/2)),2)*comb(int(np.floor(n/2)),2)))


def _diversity_ESCAPE(g):
    """
    Credit: slight modification from https://bitbucket.org/seshadhri/escape/src/master/
    """
    G=nx.complement(g)
    n=G.number_of_nodes()
    order = DegenOrdering(G)   # Get degeneracy ordering
    DG = Orient(G,order)        # DG is digraph with this orientation
    
    size = n
    tri_info = triangle_info(DG) # Get triangle info
    
    star_3 = 0.0
    path_3 = 0.0
    tailed_triangle = 0.0
    cycle_4 = 0.0
    chordal_cycle = 0.0
    clique_4 = 0.0
    
    for node in G.nodes():         
        for nbr in G[node]:             # Loop over neighbors of node
            tri_edge = tri_info[1][(node,nbr)]
            chordal_cycle += tri_edge*(tri_edge-1)/2 # Number of chordal-cycles hinged at edge e = {d_e \choose 2}     
    
    # Previous code counts each chordal-cycle twice because each edge appears twice in loop.
    chordal_cycle = chordal_cycle/2

    wedge_outout = {}       # Hash tables for storing wedges

    # The directed interpretation of Chiba-Nishizeki: for each (u,v), count the number of out wedges and in-out wedges with ends (u,v)

    # There are 3-types of directed 4-cycles
    type1 = 0.0
    type2 = 0.0
    type3 = 0.0

    outout_nz = 0.0
    inout_nz = 0.0

    outout = 0.0
    inout = 0.0
    
    cycle_4 = 0.0
    
    for node in DG.vertices:
        # First we index out-out wedges centered at node
        for (nbr1, nbr2) in itertools.combinations(DG.adj_list[node],2):    #Loop over all pairs of neighbors of node1
            if nbr1 > nbr2:     # If nbr1 > nbr2, swap, so that nbr1 \leq nbr2
                tmp = nbr1
                nbr1 = nbr2
                nbr2 = tmp            

           # print(node,nbr1,nbr2)

            if (nbr1,nbr2) in wedge_outout:    # If (nbr1,nbr2) already seen, increment wedge count
                wedge_outout[(nbr1,nbr2)] += 1
                outout += 1
            else:
                outout_nz += 1
                wedge_outout[(nbr1,nbr2)] = 1  # Else initialize wedge count to 1

    for node in DG.vertices:
        endpoints = {}
        for nbr1 in DG.adj_list[node]:
            for nbr2 in DG.adj_list[nbr1]:       # Get in-out wedge with source at node
                if nbr2 in endpoints:
                    endpoints[nbr2] += 1
                    inout += 1
                else:
                    endpoints[nbr2] = 1
                    inout_nz += 1

        for v in endpoints:
            count = endpoints[v]
            type2 += count*(count-1)/2

            v1 = node
            v2 = v

            if v1 > v2:
                swp = v1
                v1 = v2
                v2 = swp

            if (v1,v2) in wedge_outout:
                type3 += count*wedge_outout[(v1,v2)]

    for pair in wedge_outout:       # Loop over all pairs in wedge_outout
        outout += 1
        count = wedge_outout[pair]  
        type1 += count*(count-1)/2  # Number of type1 4-cycles hinged at (u,v) = {W^{++}_{u,v} \choose 2}

    cycle_4 = type1 + type2 + type3

    clique_work = 0.0
    for node in DG.vertices:        # Loop over nodes
        nbrs = DG.adj_list[node]
        nbrs_info = []
        for cand in nbrs:           # Get topological order position for each cand in nbrs
            nbrs_info.append((cand,DG.top_order_inv[cand]))

        sorted_nbrs = sorted(nbrs_info, key=lambda entry: entry[1])   # Sort nbrs according to position in topological ordering

        deg = len(sorted_nbrs)      # Out-degree of node
        for i in range(0,deg):      # Loop over neighbors in sorted order
            nbri = sorted_nbrs[i][0]
            
            # Get all vertices nbrj > nbri that form triangle with nbri
            tri_end = [] 
            for j in range(i+1,deg):   # Loop over tuple of neighbors i < j
                nbrj = sorted_nbrs[j][0]
                if G.has_edge(nbri,nbrj):
                    tri_end.append(nbrj)  # nbrj forms triangle with (node,nbri)

            # Now look for edges among pairs in tri_end, to find 4-cliques
            for (v1, v2) in itertools.combinations(tri_end,2):
                clique_work += 1
                if G.has_edge(v1,v2):
                    clique_4 += 1
     
    cycle_4_induced=cycle_4-chordal_cycle+3*clique_4 # obtained by taking inverse of matrix in sesh's repo
    return np.sqrt(cycle_4_induced/(comb(int(np.ceil(n/2)),2)*comb(int(np.floor(n/2)),2)))

### Other global metrics ###

def degree_assortativity(G):
    if G.number_of_edges()<=1:
        return 0
    else:
        score = nx.degree_pearson_correlation_coefficient(G)
        
        score = 0 if np.isnan(score) else score
        return score

def leadership(G):
    d = [val for (node, val) in G.degree()]
    N = len(d)
    d_max = max(d)
    numer = sum([d_max - x for x in d])
    denomin = (N - 2) * (N - 1)
    return (numer / denomin)

def skewness_leadership(G,node_met=nx.pagerank):
    """
    Expects node_met to be a function that outputs scores for each vertex in a dictionary.
    Our take on leadership using skewness of the scores of a given vertex based measure. 
    Large skewness means large leadership. 
    Default function is pagerank. 
    """
    return stats.skew(list(node_met(G).values()))

    
def simple_leadership(G,node_met=nx.pagerank):
    """
    Expects node_met to be a function that outputs scores for each vertex in a dictionary
    Simplified notion of leadership akin to that given in the paper, except more flexible
    in that one can use other vertex scores besides "degree". 
    Default function is pagerank. 
    """
    vals=list(node_met(G).values())
    return (max(vals)-np.mean(vals))/G.number_of_nodes()

def bonding(G):
    if has_igraph:
        T=ig.Graph.from_networkx(G)
        return T.transitivity_undirected(mode='zero')
    else:
        return nx.transitivity(G)

def average_distance(G):
    if G.number_of_edges()==0:
        return 1
    if has_igraph:
        T=ig.Graph.from_networkx(G)
        return T.average_path_length()
    else:
        return nx.average_shortest_path_length(G)

def components(G):
    return nx.number_connected_components(G)

def component_max(G):
    return len(max(nx.connected_components(G), key=len))

def component_min(G):
    return len(min(nx.connected_components(G), key=len))

def component_mean(G):
    return np.mean([len(c) for c in nx.connected_components(G)])

def class_assortativity(G):
    # PSEM CLASS 
    return nx.attribute_assortativity_coefficient(G, "PSEM_CLASS")

def p_smoothness(G):
    """
    Measure between 0 and 1 of the entropy in the connected component sizes. 
    Values closer to 0 mean vast majority of graph is a single component. 
    Values closer to 1 mean nodes are equally spread out across k, equally 
    sized connected component, for some k. 
    """
    if nx.number_connected_components(G)==1:
        return 0 #this is set by convention
    else:
        n=G.number_of_nodes()
        dist=[len(x)/n for x in nx.connected_components(G)]
        return -1*sum([x*np.log2(x) for x in dist])/np.log2(len(dist))

def algebraic_connectivity(G):
    """
    Computes regularized, normalized algebraic connectivity. 
    """
    if nx.number_of_isolates(G)==G.number_of_nodes():
        return 0
    
    #add weak, complete weighted as spectral regularizer
    n=G.number_of_nodes()
    H=nx.complete_graph(n)
    nx.relabel_nodes(H,{k:v for k,v in zip(H.nodes(),G.nodes())},copy=False)
    
    for e in H.edges():
        H.edges[e]['weight']=1/n #choosing 1/n as regularizing value
    W=nx.compose(G,H) #merges the graphs, but only takes weights of one
    
    #correct the weights
    for e in W.edges():
        if G.has_edge(*e):
            W.edges[e]['weight']=G.edges[e]['weight']+H.edges[e]['weight']
    
    #compute weighted normalized algebraic connectivity, values between
    #0 and n/n-1
    return nx.algebraic_connectivity(W,weight='weight',normalized=True,
                                    method='tracemin_lu')

