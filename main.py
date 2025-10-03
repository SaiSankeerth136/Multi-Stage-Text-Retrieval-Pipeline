import os
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

# Core IR + evaluation
from beir import util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval

# Models
from sentence_transformers import SentenceTransformer, CrossEncoder

# ANN and utilities
import faiss
import torch
from tqdm import tqdm

# BM25
from rank_bm25 import BM25Okapi

# Learning-to-Rank
import lightgbm as lgb

CONFIG = {
    "datasets": ["fiqa"],
    # Faster models with acceptable accuracy tradeoff
    "embed_model": "BAAI/bge-base-en-v1.5",  # Smaller than 'large', 3x faster
    "rerank_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",  # 6 layers vs 12, 2x faster
    
    # Reduced depths for speed
    "retrieval_k": 500,  
    "rerank_k": 100,     
    "eval_ks": [1, 3, 5, 10],
    
    # Larger batches for throughput
    "encode_batch_size": 256,    # Increased for GPU efficiency
    "rerank_batch_size": 128,    # Increased for GPU efficiency
    "rerank_query_batch": 64,    # Process more queries at once
    
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "max_docs": None,
    "max_queries": None,
    "random_seed": 42,
    "cache_dir": "cache_beir",
    "index_dir": "cache_beir/faiss",
    "emb_dir": "cache_beir/embeddings",
    "results_dir": "cache_beir/results",
    
    # Simplified pipeline
    "use_ltr": True,
    "use_bm25": True,
    "rrf_k": 60,
}


def ensure_dirs():
    Path(CONFIG["cache_dir"]).mkdir(parents=True, exist_ok=True)
    Path(CONFIG["index_dir"]).mkdir(parents=True, exist_ok=True)
    Path(CONFIG["emb_dir"]).mkdir(parents=True, exist_ok=True)
    Path(CONFIG["results_dir"]).mkdir(parents=True, exist_ok=True)


def doc_text(doc):
    title = (doc.get("title") or "").strip()
    text = (doc.get("text") or "").strip()
    return (title + "\n" + text).strip() if title else text


def subset_corpus_queries_qrels(corpus, queries, qrels, max_docs, max_queries, seed=42):
    if max_docs is None and max_queries is None:
        return corpus, queries, qrels
    rng = np.random.RandomState(seed)
    if max_docs is not None and max_docs < len(corpus):
        all_doc_ids = list(corpus.keys())
        sel_doc_ids = list(rng.choice(all_doc_ids, size=max_docs, replace=False))
        sel_doc_set = set(sel_doc_ids)
        corpus = {did: corpus[did] for did in sel_doc_ids}
    else:
        sel_doc_set = set(corpus.keys())
    filtered_qrels = {}
    for qid, rels in qrels.items():
        kept = {did: r for did, r in rels.items() if did in sel_doc_set}
        if kept:
            filtered_qrels[qid] = kept
    filtered_queries = {qid: queries[qid] for qid in filtered_qrels.keys() if qid in queries}
    if max_queries is not None and max_queries < len(filtered_queries):
        all_qids = list(filtered_queries.keys())
        sel_qids = list(rng.choice(all_qids, size=max_queries, replace=False))
        filtered_queries = {qid: filtered_queries[qid] for qid in sel_qids}
        filtered_qrels = {qid: filtered_qrels[qid] for qid in sel_qids}
    return corpus, filtered_queries, filtered_qrels


def load_or_encode_corpus_embeddings(dataset_name, corpus_texts, model):
    model_name = CONFIG['embed_model'].replace('/', '_')
    emb_path = Path(CONFIG["emb_dir"]) / f"{dataset_name}_{model_name}_corpus_emb.npy"
    if emb_path.exists():
        print(f"  Loading cached corpus embeddings...")
        corpus_embeddings = np.load(emb_path)
    else:
        print(f"  Encoding corpus ({len(corpus_texts)} docs)...")
        corpus_embeddings = model.encode(
            corpus_texts,
            batch_size=CONFIG["encode_batch_size"],
            convert_to_numpy=True,
            show_progress_bar=True,
            normalize_embeddings=True,
        ).astype("float32")
        np.save(emb_path, corpus_embeddings)
    return corpus_embeddings


def load_or_encode_query_embeddings(dataset_name, query_ids, query_texts, model):
    model_name = CONFIG['embed_model'].replace('/', '_')
    emb_path = Path(CONFIG["emb_dir"]) / f"{dataset_name}_{model_name}_query_emb.npy"
    if emb_path.exists():
        print(f"  Loading cached query embeddings...")
        query_embeddings = np.load(emb_path)
        if query_embeddings.shape[0] != len(query_texts):
            print(f"  Query count mismatch, re-encoding...")
            query_embeddings = model.encode(
                query_texts,
                batch_size=CONFIG["encode_batch_size"],
                convert_to_numpy=True,
                show_progress_bar=True,
                normalize_embeddings=True,
            ).astype("float32")
            np.save(emb_path, query_embeddings)
    else:
        print(f"  Encoding queries ({len(query_texts)} queries)...")
        query_embeddings = model.encode(
            query_texts,
            batch_size=CONFIG["encode_batch_size"],
            convert_to_numpy=True,
            show_progress_bar=True,
            normalize_embeddings=True,
        ).astype("float32")
        np.save(emb_path, query_embeddings)
    return query_embeddings


def build_or_load_faiss_index(dataset_name, corpus_embeddings):
    dim = corpus_embeddings.shape[1]
    model_name = CONFIG['embed_model'].replace('/', '_')
    idx_path = Path(CONFIG["index_dir"]) / f"{dataset_name}_{model_name}_ip.index"
    
    corpus_norm = corpus_embeddings.copy()
    faiss.normalize_L2(corpus_norm)
    
    if idx_path.exists():
        print(f"  Loading cached FAISS index...")
        index = faiss.read_index(str(idx_path))
        if index.d != dim:
            print(f"  Dimension mismatch, rebuilding...")
            index = faiss.IndexFlatIP(dim)
            index.add(corpus_norm)
            faiss.write_index(index, str(idx_path))
    else:
        print(f"  Building FAISS index...")
        index = faiss.IndexFlatIP(dim)
        index.add(corpus_norm)
        faiss.write_index(index, str(idx_path))
    
    # Move to GPU if available for faster search
    if CONFIG["device"] == "cuda":
        try:
            if faiss.get_num_gpus() > 0:
                print(f"  Moving index to GPU...")
                res = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(res, 0, index)
        except:
            print(f"  GPU not available for FAISS, using CPU")
    
    return index


def ann_search(index, query_embeddings, k):
    query_norm = query_embeddings.copy()
    faiss.normalize_L2(query_norm)
    scores, idxs = index.search(query_norm, k)
    return scores, idxs


def format_results(qids, corpus_ids, idxs, scores, topn=None):
    results = {}
    for i, qid in enumerate(qids):
        results[qid] = {}
        limit = idxs.shape[1] if topn is None else min(topn, idxs.shape[1])
        for j in range(limit):
            doc_id = corpus_ids[idxs[i, j]]
            results[qid][doc_id] = float(scores[i, j])
    return results


def tokenize_simple(text):
    return text.lower().split()


def build_bm25_index(corpus_texts):
    print("  Tokenizing corpus for BM25...")
    tokenized_corpus = [tokenize_simple(doc) for doc in corpus_texts]
    print("  Building BM25 index...")
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25


def bm25_search_batch(bm25, corpus_ids, query_texts, k):
    """Vectorized BM25 search"""
    results = {}
    print(f"  BM25 search for {len(query_texts)} queries...")
    for qtext in tqdm(query_texts, desc="BM25", disable=len(query_texts) < 10):
        tokenized_query = tokenize_simple(qtext)
        scores = bm25.get_scores(tokenized_query)
        # Use argpartition for faster top-k (no full sort needed)
        if k < len(scores):
            top_indices = np.argpartition(scores, -k)[-k:]
            top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
        else:
            top_indices = np.argsort(scores)[::-1]
        results[qtext] = {corpus_ids[idx]: float(scores[idx]) for idx in top_indices}
    return results


def reciprocal_rank_fusion(results_list, k=60):
    fused = defaultdict(lambda: defaultdict(float))
    for results in results_list:
        for qid, docs in results.items():
            sorted_docs = sorted(docs.items(), key=lambda x: x[1], reverse=True)
            for rank, (doc_id, _) in enumerate(sorted_docs, start=1):
                fused[qid][doc_id] += 1.0 / (k + rank)
    return {qid: dict(docs) for qid, docs in fused.items()}


def rerank_results_optimized(query_ids, queries, corpus, retrieval_results, 
                             cross_encoder, rerank_k, query_batch):
    """Optimized reranking with caching and larger batches"""
    reranked = {}
    
    # Pre-cache document texts to avoid repeated processing
    doc_cache = {}
    
    for start in tqdm(range(0, len(query_ids), query_batch), desc="Reranking"):
        chunk_qids = query_ids[start:start + query_batch]
        pairs, offsets, cand_lists = [], [], []
        
        for qid in chunk_qids:
            items = list(retrieval_results[qid].items())[:rerank_k]
            cand_doc_ids = [doc_id for doc_id, _ in items]
            
            # Use cache for document texts
            cand_texts = []
            for doc_id in cand_doc_ids:
                if doc_id not in doc_cache:
                    doc_cache[doc_id] = doc_text(corpus[doc_id])
                cand_texts.append(doc_cache[doc_id])
            
            qtext = queries[qid]
            begin = len(pairs)
            pairs.extend([[qtext, p] for p in cand_texts])
            end = len(pairs)
            offsets.append((qid, begin, end))
            cand_lists.append(cand_doc_ids)
        
        if not pairs:
            for qid in chunk_qids:
                reranked[qid] = {}
            continue
        
        scores = cross_encoder.predict(
            pairs,
            batch_size=CONFIG["rerank_batch_size"],
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        scores = np.array(scores, dtype=np.float32)
        
        for (qid, begin, end), cand_doc_ids in zip(offsets, cand_lists):
            chunk_scores = scores[begin:end]
            order = np.argsort(chunk_scores)[::-1]
            reranked[qid] = {}
            for idx in order:
                reranked[qid][cand_doc_ids[idx]] = float(chunk_scores[idx])
    
    return reranked


def train_ltr_model_fast(query_ids, retrieval_results, bm25_results, reranked_results, 
                        qrels, eval_ks=None):
    """Fast LTR training with early stopping"""
    X, y, group_sizes = [], [], []

    for qid in query_ids:
        cand_items = list(reranked_results.get(qid, {}).items())
        if not cand_items:
            continue
        
        dense_ranks = {doc_id: rank for rank, (doc_id, _) in 
                      enumerate(sorted(retrieval_results[qid].items(), 
                                     key=lambda x: x[1], reverse=True), start=1)}
        bm25_ranks = {doc_id: rank for rank, (doc_id, _) in 
                     enumerate(sorted(bm25_results[qid].items(), 
                                    key=lambda x: x[1], reverse=True), start=1)} if CONFIG["use_bm25"] else {}
        
        count = 0
        for doc_id, xenc_score in cand_items:
            dense_score = retrieval_results[qid].get(doc_id, 0.0)
            bm25_score = bm25_results[qid].get(doc_id, 0.0) if CONFIG["use_bm25"] else 0.0
            dense_rank = dense_ranks.get(doc_id, len(dense_ranks) + 1)
            bm25_rank = bm25_ranks.get(doc_id, len(bm25_ranks) + 1) if CONFIG["use_bm25"] else 0
            
            rr_dense = 1.0 / dense_rank if dense_rank > 0 else 0.0
            rr_bm25 = 1.0 / bm25_rank if bm25_rank > 0 else 0.0
            
            label = qrels.get(qid, {}).get(doc_id, 0)
            X.append([dense_score, bm25_score, xenc_score, rr_dense, rr_bm25])
            y.append(int(label))
            count += 1
        
        if count > 0:
            group_sizes.append(count)

    if not X or not group_sizes:
        raise ValueError("No training instances for LTR.")

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int32)
    group_arr = np.asarray(group_sizes, dtype=np.int32)

    # Faster LTR with fewer estimators and early stopping
    ltr_model = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=100,  # Reduced from 500
        learning_rate=0.1,  # Increased for faster convergence
        num_leaves=31,
        max_depth=6,  # Reduced depth
        random_state=CONFIG["random_seed"],
        n_jobs=-1,
        verbose=-1,
    )

    fit_kwargs = {}
    if eval_ks is not None:
        fit_kwargs["eval_at"] = list(eval_ks)

    ltr_model.fit(X, y, group=group_arr, **fit_kwargs)
    return ltr_model


def apply_ltr_rerank(ltr_model, query_ids, retrieval_results, bm25_results, reranked_results):
    out = {}
    for qid in query_ids:
        cand_items = list(reranked_results.get(qid, {}).items())
        if not cand_items:
            out[qid] = {}
            continue
        
        dense_ranks = {doc_id: rank for rank, (doc_id, _) in 
                      enumerate(sorted(retrieval_results[qid].items(), 
                                     key=lambda x: x[1], reverse=True), start=1)}
        bm25_ranks = {doc_id: rank for rank, (doc_id, _) in 
                     enumerate(sorted(bm25_results[qid].items(), 
                                    key=lambda x: x[1], reverse=True), start=1)} if CONFIG["use_bm25"] else {}
        
        doc_ids = [doc_id for doc_id, _ in cand_items]
        X = []
        for doc_id, xenc_score in cand_items:
            dense_score = retrieval_results[qid].get(doc_id, 0.0)
            bm25_score = bm25_results[qid].get(doc_id, 0.0) if CONFIG["use_bm25"] else 0.0
            dense_rank = dense_ranks.get(doc_id, len(dense_ranks) + 1)
            bm25_rank = bm25_ranks.get(doc_id, len(bm25_ranks) + 1) if CONFIG["use_bm25"] else 0
            rr_dense = 1.0 / dense_rank if dense_rank > 0 else 0.0
            rr_bm25 = 1.0 / bm25_rank if bm25_rank > 0 else 0.0
            X.append([dense_score, bm25_score, xenc_score, rr_dense, rr_bm25])
        
        X = np.asarray(X, dtype=np.float32)
        preds = ltr_model.predict(X)
        order = np.argsort(preds)[::-1]
        ranked = {doc_ids[idx]: float(preds[idx]) for idx in order}
        out[qid] = ranked
    return out


def evaluate(qrels, results, ks):
    evaluator = EvaluateRetrieval()
    return evaluator.evaluate(qrels, results, k_values=ks)


def main():
    ensure_dirs()
    
    print(f"Device: {CONFIG['device']}")
    print(f"Loading embedding model: {CONFIG['embed_model']}")
    embed_model = SentenceTransformer(CONFIG["embed_model"], device=CONFIG["device"])
    
    print(f"Loading cross-encoder: {CONFIG['rerank_model']}")
    cross_encoder = CrossEncoder(CONFIG["rerank_model"], device=CONFIG["device"])
    
    all_results = {}

    for dataset_name in CONFIG["datasets"]:
        print(f"\n{'='*60}")
        print(f"Processing {dataset_name.upper()}")
        print(f"{'='*60}")
        
        url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
        data_path = util.download_and_unzip(url, "datasets")
        corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")
        print(f"Dataset: Corpus={len(corpus)} | Queries={len(queries)}")
        
        corpus, queries, qrels = subset_corpus_queries_qrels(
            corpus, queries, qrels,
            max_docs=CONFIG["max_docs"],
            max_queries=CONFIG["max_queries"],
            seed=CONFIG["random_seed"],
        )

        query_ids = [qid for qid in queries.keys() if qid in qrels]
        if not query_ids:
            print("No labeled queries; skipping.")
            continue

        corpus_ids = list(corpus.keys())
        corpus_texts = [doc_text(corpus[doc_id]) for doc_id in corpus_ids]
        query_texts = [queries[qid] for qid in query_ids]

        # Stage 1: Dense retrieval
        print("\n[1/5] Dense retrieval")
        corpus_embeddings = load_or_encode_corpus_embeddings(dataset_name, corpus_texts, embed_model)
        query_embeddings = load_or_encode_query_embeddings(dataset_name, query_ids, query_texts, embed_model)
        index = build_or_load_faiss_index(dataset_name, corpus_embeddings)

        k = min(CONFIG["retrieval_k"], len(corpus_ids))
        print(f"  Searching top-{k}...")
        scores, idxs = ann_search(index, query_embeddings, k)
        dense_results = format_results(query_ids, corpus_ids, idxs, scores, topn=k)

        # Stage 2: BM25
        print("\n[2/5] BM25 retrieval")
        bm25 = build_bm25_index(corpus_texts)
        bm25_raw = bm25_search_batch(bm25, corpus_ids, query_texts, k)
        bm25_results = {qid: bm25_raw[queries[qid]] for qid in query_ids}

        # Stage 3: Hybrid fusion
        print("\n[3/5] Hybrid RRF fusion")
        retrieval_results = reciprocal_rank_fusion([dense_results, bm25_results], k=CONFIG["rrf_k"])

        # Stage 4: Reranking
        r_k = min(CONFIG["rerank_k"], k)
        print(f"\n[4/5] Reranking top-{r_k}")
        reranked_results = rerank_results_optimized(
            query_ids=query_ids,
            queries=queries,
            corpus=corpus,
            retrieval_results=retrieval_results,
            cross_encoder=cross_encoder,
            rerank_k=r_k,
            query_batch=CONFIG["rerank_query_batch"],
        )

        # Stage 5: LTR
        if CONFIG["use_ltr"]:
            print(f"\n[5/5] LTR training and application")
            ltr_model = train_ltr_model_fast(
                query_ids=query_ids,
                retrieval_results=dense_results,
                bm25_results=bm25_results,
                reranked_results=reranked_results,
                qrels=qrels,
                eval_ks=CONFIG["eval_ks"],
            )

            reranked_results = apply_ltr_rerank(
                ltr_model=ltr_model,
                query_ids=query_ids,
                retrieval_results=dense_results,
                bm25_results=bm25_results,
                reranked_results=reranked_results,
            )

        # Evaluation
        print("\n" + "="*60)
        print("EVALUATION")
        print("="*60)
        
        ndcg_dense, _, _, _ = evaluate(qrels, dense_results, CONFIG["eval_ks"])
        ndcg_hybrid, _, _, _ = evaluate(qrels, retrieval_results, CONFIG["eval_ks"])
        ndcg_final, map_final, recall_final, precision_final = evaluate(qrels, reranked_results, CONFIG["eval_ks"])

        print(f"\nDense-only:  NDCG@10 = {ndcg_dense['NDCG@10']:.4f}")
        print(f"Hybrid:      NDCG@10 = {ndcg_hybrid['NDCG@10']:.4f}")
        print(f"Final:       NDCG@10 = {ndcg_final['NDCG@10']:.4f}")
        print(f"             NDCG@1  = {ndcg_final['NDCG@1']:.4f}")
        print(f"             NDCG@3  = {ndcg_final['NDCG@3']:.4f}")
        print(f"             NDCG@5  = {ndcg_final['NDCG@5']:.4f}")

        improvement = ndcg_final["NDCG@10"] - ndcg_dense["NDCG@10"]
        pct = (improvement / ndcg_dense["NDCG@10"]) * 100.0 if ndcg_dense["NDCG@10"] > 0 else 0.0
        print(f"\nImprovement: +{improvement:.4f} ({pct:.1f}%)")

        all_results[dataset_name] = {
            "dense": ndcg_dense["NDCG@10"],
            "final": ndcg_final["NDCG@10"],
            "improvement": float(improvement),
        }

        out_path = Path(CONFIG["results_dir"]) / f"{dataset_name}_results.json"
        with open(out_path, "w") as f:
            json.dump(all_results[dataset_name], f, indent=2)

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for ds, res in all_results.items():
        print(f"{ds.upper()}: {res['dense']:.4f} → {res['final']:.4f} (+{res['improvement']:.4f})")


if __name__ == "__main__":
    main()
