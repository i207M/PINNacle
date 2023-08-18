#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 12:49:24 2021

@author: bmoseley
"""

# This module defines the base class used by domains.py 
# It defines a ND domain with overlapping rectangular subdomains

# This class is used by domains.py

import itertools
from bisect import bisect_left
    
import numpy as np


class _RectangularDomainND:
    """
    Base class which defines a ND domain with hyperrectangular subdomains
    FBPINN uses ActiveRectangularDomainND inherited from this class; PINN does not use this class.
    """
    
    def __init__(self, subdomain_xs, subdomain_ws):
        """Creates hyperrectangle subdomains with spacing subdomain_xs
        and overlap width subdomain_ws.
        subdomain_xs: list of rectangle edges along each dimension
        subdomain_ws: list of overlap widths along each dimension
        """
        
        ## VALIDATION

        if len(subdomain_xs) != len(subdomain_ws): #check if dimensions match (subdomain_xs is like [np.array, np.array, ...])
            raise Exception("ERROR: lengths of subdomain_xs (%s) and subdomain_ws do not match (%s)!"%(len(subdomain_xs), len(subdomain_ws)))
        
        for i in range(len(subdomain_xs)): #check if #partitions of each dimension match
            if len(subdomain_xs[i]) != len(subdomain_ws[i]):
                raise Exception("ERROR: length of subdomain_x does not equal length of subdomain_w at index %i"%(i))
        
        subdomain_xs = [np.array(x).copy() for x in subdomain_xs]
        subdomain_ws = [np.array(w).copy() for w in subdomain_ws]
        
        ## PREPARATION
        
        # get dimensions
        nd = len(subdomain_xs)# NUMBER OF DIMENSIONS
        nm = tuple([len(x)-1 for x in subdomain_xs])# Number of Middle Segments / Subdomains / Models IN EACH DIMENSION
        # for example, for a division like 田, nd=2, nm=(2,2), for a division like 目, nd=2, nm=(1,3)
        
        # make widths zero on boundaries
        for w in subdomain_ws: w[-1] = w[0] = 0
        # in other words, the "overlap width" at the left and right end point is set to 0
        # for example, let subdomain_xs = [np.array([-1.0, -0.5, 0, 0.5, 1])], subdomain_ws = [array([0.2, 0.2, 0.2, 0.2, 0.2])] (width=0.4)
        # after this operation, subdomain_ws = [array([0, 0.2, 0.2, 0.2, 0]], meaning that the domain be like
        # ws  |       0     |<- 0.2->| |<- 0.2->| |<- 0.2->|      0  
        # xs  |     -1.0 ----- -0.5 -----  0.  ----- +0.5 ----- +1.0 
        # segs|          ----=      =---=      =---=      =----          | ----:non-overlap, ====:overlap
        
        # expand out xs and ws
        xs = np.meshgrid(*subdomain_xs, indexing="ij")# nd x [(nm,)+1]
        # Example: for a division like 田 where nd=2;nm=(2,2)
        # xs = [array([[x1,x1,x1],[x2,x2,x2],[x3,x3,x3]]), array([[y1,y2,y3],[y1,y2,y3],[y1,y2,y3]])]
        # shape: 2 x [(2,2)+1] = 2 x [(3,3)]
        ws = np.meshgrid(*subdomain_ws, indexing="ij")# nd x [(nm,)+1]
        # Example usage of meshgrid:
        # x=np.linspace(0,1000,20); y=np.linspace(0,500,20); X,Y=np.meshgrid(x,y); plt.plot(X,Y,color='b',marker='.');
        
        # expand out iis and make initial maps
        iis = np.meshgrid(*[range(n) for n in nm], indexing="ij")# nd x [(nm,)]
        sm0 = ms0 = np.expand_dims(np.stack(iis, 0), 0)# (1,nd,nm)
        #for example, nm=(2,3), iis=[arr([[0,0,0],[1,1,1]]), arr([[0,1,2],[0,1,2]])],
        #np.stack(iis,0) is arr([[[0,0,0],[1,1,1]], [[0,1,2],[0,1,2]]]), ms0 is arr([[..., ...]])

        # convenience slicer class
        class NDSlicer:
            def __getitem__(self, s):
                return (s,)*nd
                # expands to nd dimensions, returns (s,s,...,s) where s appears nd times
                # Example sl[1] = (1,1), sl[1:] = (slice(1,None,None), slice(1,None,None)), '1:' is interpreted as a slice
                # ls = [1,2,3,4,5], then ls[slice(1,None,None)]==ls[1:]==[2,3,4,5], ls[slice(1,None,2)]==ls[1::2]==[2,4]
                # let nd=2
                # mt = np.array([[1,2,3],[4,5,6],[7,8,9]]), then mt[sl[1:]]==mt[(slice, slice)]==mt[1:,1:]==[[5,6],[8,9]]
            def cut_edges(self, constrained_axes):
                return tuple([slice(None)]*2+[slice(None, -1) if i in constrained_axes else slice(None) for i in range(nd)])
                # cuts edges of constrained axes
                # let nd = 3
                # sl.cut_edges(()) == (slice(None),...,slice(None)) where slice(None) appears 5 times
                # sl.cut_edges((0,1)) == (slice(None),slice(None),slice(None,-1,None),slice(None,-1,None),slice(None))
        sl = NDSlicer()
        
        
        
        ## COMPUTE SEGMENTS / MAPS

        # NOTE: models_segments will contain segment indices outside of grid (in order to maintain tensor structure)
        # these need to be discarded in outside code
        
        segments = []
        segments_models = []
        models_segments = []
        ca2idx = dict()
        
        # for increasing numbers of constrained axes
        for order in range(0, nd+1):
            
            # for each set of constrained axis combinations
            for constrained_axes in itertools.combinations(range(nd), order):
                # Example: nd=3 => constrained_axes is () (0,) (1,) (2,) (0,1,) (0,2,) (1,2,) (0,1,2,)
                ca2idx[constrained_axes] = len(segments)
                
                # SEGMENTS
                
                # (2,nd)+nd is a tuple. Example: (1,2)+(3,) = (1,2,3)
                s = np.zeros((2,nd)+nm)# (2,nd,nm)    (left or right, dim, grid) xyz
                
                # for each dimension
                for i in range(nd):
                    if i in constrained_axes:# if this dimension is constrained
                        # xn - wn/2, xn + wn/2 (illustration: ====, overlap)
                        s[0,i], s[1,i] = xs[i][sl[1:]]-ws[i][sl[1:]]/2, xs[i][sl[1:]]+ws[i][sl[1:]]/2
                        # should be xs[i][sl[1:-1]]\pm ws[i][sl[1:-1]]/2, but this would give (nm,)-1, inconsistent with shape of s
                    else:
                        # x{n-1} + w{n-1}/2, xn - wn/2 (illustration: ----, non-overlap)
                        s[0,i], s[1,i] = xs[i][sl[:-1]]+ws[i][sl[:-1]]/2, xs[i][sl[1:]]-ws[i][sl[1:]]/2
                s = s[sl.cut_edges(constrained_axes)]# cut constrained edges so that s[0/1,i] actually be xs[i][sl[1:-1]]\pm ws[i][sl[1:-1]]/2
                segments.append(s) #now s shape is (2,nd,nm') where nm' is nm after sl.cut_edges
                
                # MAPS
                
                n_overlap = 2**len(constrained_axes)# number of overlapping models
                sm = np.concatenate([sm0.copy() for _ in range(n_overlap)])# (ne,nd,nm)    (model, dim, grid) ijk
                ms = np.concatenate([ms0.copy() for _ in range(n_overlap)])# (ne,nd,nm)    (segment, dim, grid) ijk
                
                # for all overlapping elements
                for iel,offsets in enumerate(itertools.product(*([[0,1]]*len(constrained_axes)))):
                # let len(constrained_axes) be 2, then iel, offsets are (iel=i element)
                # 0, (0,0); 1, (0,1); 2, (1,0); 3, (1,1);
                    # for each constrained dimension
                    for ic,offset in enumerate(offsets):# add their offsets to map
                        #offset is either 0 or 1 (broadcast)
                        sm[iel,constrained_axes[ic]] += offset
                        ms[iel,constrained_axes[ic]] -= offset
                sm = sm[sl.cut_edges(constrained_axes)]# cut constrained edges
                segments_models.append(sm)
                models_segments.append(ms)
                # concerning this iteration's s, sm, ms
                # for example, if nm=(2,3), constrained_axes=(0,1), then
                # sm[:,:,0,0] = [[0,0], [0,1], [1,0], [1,1]], segment [0,0] maps to models [0,0],[0,1],[1,0],[1,1]
                # sm[:,:,0,1] = [[0,1], [0,2], [1,1], [1,2]], segment [0,1] maps to models [0,1],[0,2],[1,1],[1,2]
                # +--+--+
                # +--+--+
                # +--+--+
                # +--+--+
                
            
        ## SAVE SELF ATTRIBUTES
        
        self.nd = nd
        self.nm = nm
        self.segments = segments
        self.segments_models = segments_models
        self.models_segments = models_segments
        
        
        ## FINAL VALIDATION
        
        for ioa,s in enumerate(self.segments):
            if np.any(s[0] > s[1]):
                raise Exception("ERROR: segments are negative! (%i)"%(ioa))
            
        
        # HELPERS
        
        self.xx = np.stack(xs, 0)# (nd,nm+1), example 田: xs.shape=2x[(3,3)], xx.shape=(2, 3, 3)
        self.ww = np.stack(ws, 0)# (nd,nm+1)
        
        # helper for determining which segment a point belongs to (omn)

        self.xmarks = list()
        for ithd in range(self.nd):
            x = subdomain_xs[ithd]
            w = subdomain_ws[ithd]
            # x0+w0/2, x1-w1/2, x1+w1/2, ..., xn-wn/2
            xmark = []
            for xx, ww in zip(x,w):
                xmark.append(xx-ww/2); xmark.append(xx+ww/2)
            self.xmarks.append(xmark[1:-1])
        self.ca2idx = ca2idx
        
    def get_onm(self, x):
        constrained_axes = tuple()
        dim_index = [None for _ in range(self.nd)]
        for ithd in range(self.nd):
            pos = bisect_left(self.xmarks[ithd], x[ithd])-1
            pos = min(max(pos, 0), len(self.xmarks[ithd])-2)
            # x[ithd] is in the pos-th interval of xmark[ithd]
            if pos % 2 == 0: # dim is not constrained
                dim_index[ithd] = pos // 2
            else: # dim is constrained
                constrained_axes = constrained_axes + (ithd,)
                dim_index[ithd] = pos // 2
        return (self.ca2idx[constrained_axes],) + tuple(dim_index)

    def __str__(self):

        st  = "nd:"+str(self.nd)+"\n"
        st += "nm:"+str(self.nm)+"\n"
        
        st += "segments:"+"\n"
        for s in self.segments:
            st += str(s.shape)+" "+str(s)+"\n"
        
        st += "segments_models:"+"\n"
        for sm in self.segments_models:
            st += str(sm.shape)+" "+str(sm)+"\n"
        
        st += "models_segments:"+"\n"
        for ms in self.models_segments:
            st += str(ms.shape)+" "+str(ms)+"\n"
            
        return st[:-1]
            
            
if __name__ == "__main__":
    
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']*100
    itergrid = lambda shape: enumerate(itertools.product(*[np.arange(d) for d in shape]))
    
    
    # 1D test
    
    for subdomain_xs, subdomain_ws in [
                                       [[np.array([-5,0,3,6,10])], 
                                        [0.2*np.array([1,2,3,4,5])]],
                                       
                                       [[np.array([-5,0])], 
                                        [0.2*np.array([1,2])]],
                                       ]:
        
        D = _RectangularDomainND(subdomain_xs, subdomain_ws)
        print(D)
        
        ss,xx = D.segments, D.xx
        plt.figure(figsize=(8*len(ss), 6))
        for iplot,(s,sm,ms) in enumerate(zip(D.segments, D.segments_models, D.models_segments)):
            # create a sub-plot for each combination of constrained_axes
            ax = plt.subplot(1,len(ss),iplot+1)
            
            # plot subdomains (same for every subplot)
            for im,(i,) in itergrid(D.nm):# for each subdomain, im is a counter like 0,1,2... , (i,) are the indices
                plt.plot([subdomain_xs[0][i], subdomain_xs[0][i+1]],[0,0], color=colors[im])
            
            # plot segments (for this combination of axes)
            for iseg,(i,) in itergrid(s.shape[2:]):# for each segment
                plt.plot([s[0,0,i], s[1,0,i]],[0.1,0.1], color=colors[iseg]) #shape of s: (left or right, which dim, grid)
            
            # plot maps (for this combination of axes)
            for iel in range(sm.shape[0]): #iel: ith element
                for iseg,(i,) in itergrid(sm.shape[2:]):
                    plt.scatter(xx[0,i]+sm[iel,0,i], [0.2,], c=colors[iseg], s=80) #dot is located sm[iel,0,i] units right to the left endpoint
                    
            for iel in range(ms.shape[0]):
                for iseg,(i,) in itergrid(ms.shape[2:]):
                    plt.scatter(xx[0,i]+ms[iel,0,i], [0.3,], c=colors[iseg], s=40, linewidths=1, edgecolor="k")
                
            #ax.set_aspect("equal")
            plt.autoscale()
        plt.show()
        
    
    # 2D test
    
    for subdomain_xs, subdomain_ws in [
                                       [[np.array([-5,0,3,6,10]),np.array([5,15,35])], 
                                        [0.2*np.array([1,2,3,4,5]),0.2*np.array([5,6,7])]],
                                       
                                       [[np.array([-5,0]),np.array([5,15,35])], 
                                        [0.2*np.array([1,2]),0.2*np.array([5,6,7])]],
                                       ]:
        
        D = _RectangularDomainND(subdomain_xs, subdomain_ws)
        print(D)
        
        ss,xx = D.segments, D.xx
        plt.figure(figsize=(4*len(ss), 10))
        for iplot,(s,sm,ms) in enumerate(zip(D.segments, D.segments_models, D.models_segments)):
            ax = plt.subplot(1,len(ss),iplot+1)
            
            # plot subdomains
            for im,(i,j) in itergrid(D.nm):# for each subdomain
                rect = patches.Rectangle((subdomain_xs[0][i], subdomain_xs[1][j]), #xy
                                         subdomain_xs[0][i+1]-subdomain_xs[0][i], #width
                                         subdomain_xs[1][j+1]-subdomain_xs[1][j], #height
                                         linewidth=1, edgecolor=colors[im], facecolor='none')
                ax.add_patch(rect)
            
            # plot segments
            for iseg,(i,j) in itergrid(s.shape[2:]):# for each segment
                rect = patches.Rectangle((s[0,0,i,j], s[0,1,i,j]), #xy (x,y)
                                         s[1,0,i,j]-s[0,0,i,j], #width (along x)
                                         s[1,1,i,j]-s[0,1,i,j], #height (along y)
                                         linewidth=2, edgecolor=colors[iseg], facecolor='none')
                ax.add_patch(rect)
            
            # plot maps
            for iel in range(sm.shape[0]):
                for iseg,(i,j) in itergrid(sm.shape[2:]):
                    plt.scatter(xx[0,i,j]+sm[iel,0,i,j], xx[1,i,j]+sm[iel,1,i,j], c=colors[iseg], s=80)
                    
            for iel in range(ms.shape[0]):
                for iseg,(i,j) in itergrid(ms.shape[2:]):
                    plt.scatter(xx[0,i,j]+ms[iel,0,i,j], xx[1,i,j]+ms[iel,1,i,j], c=colors[iseg], s=40, linewidths=1, edgecolor="k")
                
            ax.set_aspect("equal")
            plt.autoscale()
        plt.show()
        
    
    # 3D test
    
    for subdomain_xs, subdomain_ws in [
                                       [[np.array([-5,0,3,6,10]),np.array([5,15,35,45]),np.array([-2,5,12])],
                                        [0.2*np.array([1,2,3,4,5]),0.2*np.array([2,3,4,5]),0.2*np.array([4,5,6])]],
                                       
                                       [[np.array([-5,0]),np.array([5,15,35,45]),np.array([-2,5])],
                                        [0.2*np.array([1,2]),0.2*np.array([2,3,4,5]),0.2*np.array([4,5])]],
                                       ]:
        
        D = _RectangularDomainND(subdomain_xs, subdomain_ws)
        print(D)
    
        ss,xx = D.segments, D.xx
        plt.figure(figsize=(4*len(ss), 10))
        for iplot,(s,sm,ms) in enumerate(zip(D.segments, D.segments_models, D.models_segments)):
            ax = plt.subplot(1,len(ss),iplot+1)
            
            # plot subdomains
            for im,(i,j,k) in itergrid(D.nm):# for each subdomain
                rect = patches.Rectangle((subdomain_xs[0][i], subdomain_xs[1][j]), #xy
                                         subdomain_xs[0][i+1]-subdomain_xs[0][i], #width
                                         subdomain_xs[1][j+1]-subdomain_xs[1][j], #height
                                         linewidth=1, edgecolor=colors[im], facecolor='none')
                ax.add_patch(rect)
            
            # plot segments
            for iseg,(i,j,k) in itergrid(s.shape[2:]):# for each segment
                rect = patches.Rectangle((s[0,0,i,j,k], s[0,1,i,j,k]), #xy (x,y)
                                         s[1,0,i,j,k]-s[0,0,i,j,k], #width (along x)
                                         s[1,1,i,j,k]-s[0,1,i,j,k], #height (along y)
                                         linewidth=2, edgecolor=colors[iseg], facecolor='none')
                ax.add_patch(rect)
            
            # plot maps
            for iel in range(sm.shape[0]):
                for iseg,(i,j,k) in itergrid(sm.shape[2:]):
                    plt.scatter(xx[0,i,j,k]+sm[iel,0,i,j,k], xx[1,i,j,k]+sm[iel,1,i,j,k], c=colors[iseg], s=80)
                    
            for iel in range(ms.shape[0]):
                for iseg,(i,j,k) in itergrid(ms.shape[2:]):
                    plt.scatter(xx[0,i,j,k]+ms[iel,0,i,j,k], xx[1,i,j,k]+ms[iel,1,i,j,k], c=colors[iseg], s=40, linewidths=1, edgecolor="k")
                
            ax.set_aspect("equal")
            plt.autoscale()
        plt.show()
    
    







