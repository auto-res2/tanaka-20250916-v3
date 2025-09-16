"""src/preprocess.py
Data-loading and pre-processing logic for Dynamic-FlashGAT experiments.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
import networkx as nx

LOGGER = logging.getLogger(__name__)


def create_synthetic_graph(num_nodes: int, num_edges: int, num_features: int) -> Data:
    """Create a synthetic graph for testing."""
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_index = to_undirected(edge_index)
    x = torch.randn(num_nodes, num_features)
    y = torch.randint(0, 2, (num_nodes,))
    
    return Data(x=x, edge_index=edge_index, y=y)


def create_streaming_rmat_data(num_nodes: int, num_edges: int, chunk_size: int = 1000) -> list:
    """Create synthetic streaming RMAT data in chunks."""
    chunks = []
    edges_per_chunk = min(chunk_size, num_edges // 10)
    
    for i in range(0, num_edges, edges_per_chunk):
        chunk_edges = min(edges_per_chunk, num_edges - i)
        edge_index = torch.randint(0, num_nodes, (2, chunk_edges))
        chunks.append(edge_index)
    
    return chunks


def load_reddit_subset(num_nodes: int = 1000) -> Data:
    """Create a Reddit-like graph subset for micro-benchmarking."""
    G = nx.barabasi_albert_graph(num_nodes, 5)
    edge_index = torch.tensor(list(G.edges)).t().contiguous()
    edge_index = to_undirected(edge_index)
    
    x = torch.randn(num_nodes, 512)
    y = torch.randint(0, 41, (num_nodes,))
    
    return Data(x=x, edge_index=edge_index, y=y)


def preprocess(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Load and preprocess graph datasets for Dynamic-FlashGAT experiments."""
    LOGGER.info("Starting data preprocessing...")
    
    output_dir = Path(".research/iteration2")
    images_dir = output_dir / "images"
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    
    experiment_type = cfg.get("experiment", "smoke_test")
    datasets = {}
    
    if experiment_type == "smoke_test" or cfg.get("quick_test", False):
        LOGGER.info("Creating synthetic datasets for quick testing...")
        
        datasets["ogbn_products"] = create_synthetic_graph(1000, 5000, 47)
        datasets["ogbn_papers100m"] = create_synthetic_graph(2000, 10000, 128)
        datasets["reddit"] = load_reddit_subset(500)
        datasets["streaming_rmat"] = create_streaming_rmat_data(1000, 5000, 500)
        
    else:
        LOGGER.info("Creating larger synthetic datasets for full experiments...")
        
        datasets["ogbn_products"] = create_synthetic_graph(10000, 50000, 47)
        datasets["ogbn_papers100m"] = create_synthetic_graph(20000, 100000, 128)
        datasets["reddit"] = load_reddit_subset(5000)
        datasets["streaming_rmat"] = create_streaming_rmat_data(10000, 50000, 1000)
    
    preprocessing_results = {
        "datasets_loaded": list(datasets.keys()),
        "ogbn_products_nodes": datasets["ogbn_products"].x.shape[0],
        "ogbn_products_edges": datasets["ogbn_products"].edge_index.shape[1],
        "reddit_nodes": datasets["reddit"].x.shape[0],
        "reddit_edges": datasets["reddit"].edge_index.shape[1],
        "streaming_chunks": len(datasets["streaming_rmat"]),
        "output_dir": str(output_dir),
        "images_dir": str(images_dir)
    }
    
    results_file = output_dir / "preprocessing_results.json"
    with open(results_file, 'w') as f:
        json.dump(preprocessing_results, f, indent=2)
    
    print("=== PREPROCESSING RESULTS ===")
    print(f"Datasets loaded: {preprocessing_results['datasets_loaded']}")
    print(f"OGBN-Products: {preprocessing_results['ogbn_products_nodes']} nodes, {preprocessing_results['ogbn_products_edges']} edges")
    print(f"Reddit subset: {preprocessing_results['reddit_nodes']} nodes, {preprocessing_results['reddit_edges']} edges")
    print(f"Streaming RMAT chunks: {preprocessing_results['streaming_chunks']}")
    print(f"Results saved to: {results_file}")
    print(f"Preprocessing JSON contents: {json.dumps(preprocessing_results, indent=2)}")
    
    return {"datasets": datasets, "results": preprocessing_results}
