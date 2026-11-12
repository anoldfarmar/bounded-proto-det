"""TokenCut-style NCut on fixed DINOv2 patch features, not a TokenCut reproduction.

Keeps the published max-|eigenvector| seed / 4-connected-component rule.
Uses a single threshold calibrated on 50 normal references, rather than assuming
the original DINO key-feature threshold transfers to intermediate DINOv2 tokens.
"""
import numpy as np
from scipy import ndimage
from scipy.linalg import eigh
from threadpoolctl import threadpool_limits


def cosine_graph(features):
    features=np.asarray(features,dtype=np.float64)
    if features.ndim!=2 or not np.isfinite(features).all():raise ValueError('Invalid patch features')
    lengths=np.linalg.norm(features,axis=1,keepdims=True)
    if (lengths<1e-12).any():raise ValueError('Zero-norm patch feature')
    with threadpool_limits(limits=1):
        normalized=features/lengths
        similarity=np.clip(normalized@normalized.T,-1.,1.)
    return (similarity+similarity.T)*.5


def solve_graph(affinity):
    """Solve (D-A)v=lambda Dv via the equivalent symmetric normalized Laplacian."""
    degree=affinity.sum(axis=1)
    inv_sqrt=1./np.sqrt(degree)
    laplacian=np.eye(len(degree))-inv_sqrt[:,None]*affinity*inv_sqrt[None,:]
    laplacian=(laplacian+laplacian.T)*.5
    with threadpool_limits(limits=1):
        values,vectors=eigh(laplacian,subset_by_index=[1,2],driver='evr',check_finite=False)
    # u=D^(1/2)v, so convert back before partitioning and automatic seed selection.
    eigenvector=vectors[:,0]*inv_sqrt
    if not np.isfinite(eigenvector).all():raise ValueError('Invalid eigenvector')
    return values,eigenvector


def select_regions(eigenvector,grid,cfg):
    vector=np.asarray(eigenvector,dtype=np.float64).reshape(grid)
    positive=vector>vector.mean()
    center=np.zeros(grid,bool);b=cfg['center_border_fraction']
    center[int(grid[0]*b):int(grid[0]*(1-b)),int(grid[1]*b):int(grid[1]*(1-b))]=True
    qp=float(positive[center].mean()-positive[~center].mean())
    qn=-qp
    seed=int(np.argmax(np.abs(vector)))
    seed_side=positive if positive.flat[seed] else ~positive
    structure=ndimage.generate_binary_structure(2,1)

    def component(side,anchor):
        labels,_=ndimage.label(side,structure=structure)
        component_id=labels.flat[anchor]
        if component_id==0:raise ValueError('Seed outside selected partition')
        mask=labels==component_id;area=float(mask.mean());fallback=None
        if area<cfg['min_area_fraction']:fallback='component_too_small'
        elif area>cfg['max_area_fraction']:fallback='component_near_full'
        info=dict(seed_index=anchor,partition_area_fraction=float(side.mean()),raw_area_fraction=area,fallback=fallback)
        return (np.ones(grid,bool) if fallback else mask),info

    main,main_info=component(seed_side,seed)
    if abs(qp-qn)<=cfg['tie_tolerance']:
        diagnostic=np.ones(grid,bool)
        center_info=dict(seed_index=-1,partition_area_fraction=1.,raw_area_fraction=1.,fallback='center_tie')
    else:
        side=positive if qp>qn else ~positive
        anchor=int(np.argmax(np.where(side,np.abs(vector),-np.inf)))
        diagnostic,center_info=component(side,anchor)
    return main,diagnostic,dict(main=main_info,center=center_info,
                                positive_enrichment=qp,negative_enrichment=qn),positive


def ncut_regions(features,grid,cfg,threshold):
    similarity=cosine_graph(features)
    n=len(similarity)
    if n!=int(np.prod(grid)) or n<3:raise ValueError('Invalid graph/grid size')
    upper=similarity[np.triu_indices(n,1)]
    positive=similarity>threshold
    density=float((upper>threshold).mean())
    fallback=None
    if np.ptp(upper)<1e-10:fallback='constant_graph'
    elif density<cfg['min_graph_density']:fallback='graph_too_sparse'
    elif density>cfg['max_graph_density']:fallback='graph_near_complete'
    if fallback:
        info=dict(main=dict(seed_index=-1,raw_area_fraction=1.,partition_area_fraction=1.,fallback=fallback),
                  center=dict(seed_index=-1,raw_area_fraction=1.,partition_area_fraction=1.,fallback=fallback),
                  graph_density=density,lambda2=None,lambda3=None,eigen_gap=None,
                  positive_enrichment=None,negative_enrichment=None)
        return np.ones(grid,bool),np.ones(grid,bool),info,np.zeros(grid,bool)
    affinity=np.where(positive,1.,cfg['epsilon'])
    np.fill_diagonal(affinity,1.)
    values,vector=solve_graph(affinity)
    main,center,info,partition=select_regions(vector,grid,cfg)
    gap=float(values[1]-values[0])
    if gap<cfg['min_eigen_gap']:
        main=center=np.ones(grid,bool)
        for key in ('main','center'):info[key]['fallback']='ambiguous_eigenvector'
    info.update(graph_density=density,lambda2=float(values[0]),lambda3=float(values[1]),eigen_gap=gap)
    return main,center,info,partition
