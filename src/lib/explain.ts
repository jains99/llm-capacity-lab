export type Explainer = {
  title: string;
  kicker: string;
  body: string[];
  formula?: string;
  source?: { label: string; href: string };
};

export const EXPLAINERS: Record<string, Explainer> = {
  vram: {
    title: "Where the VRAM goes",
    kicker: "Memory",
    body: [
      "A serving GPU holds four things: the model weights, the KV cache for every in-flight request, runtime workspace, and a safety margin for fragmentation.",
      "Weights are fixed. KV cache scales with users × context. That is why the same model can serve 200 people at 2k context or 4 people at 128k on the same hardware.",
      "The 8% haircut in this simulator stands in for CUDA context, graph captures, allocator fragmentation and activations. Real fleets leave similar headroom.",
    ],
    formula: "usable = VRAM × 0.92\nused = weights + KV × users + runtime",
  },
  kv: {
    title: "The KV cache is the context",
    kicker: "Mechanism",
    body: [
      "Every generated token attends to every previous token. To avoid recomputing the past, the model stores the Key and Value vectors for each layer and each token. That store is the KV cache.",
      "It grows linearly with context and with the number of concurrent requests, and it lives in HBM next to the weights. When people say 'context is expensive', this is the bill.",
      "Grouped-query attention (GQA) is why modern models are affordable: 8 KV heads instead of 64 cut this cache by 8×.",
    ],
    formula: "KV/token = 2 × layers × kv_heads × head_dim × bytes",
    source: {
      label: "vLLM: PagedAttention paper",
      href: "https://arxiv.org/abs/2309.06180",
    },
  },
  users: {
    title: "Users are not requests",
    kicker: "Capacity",
    body: [
      "Users arrive, think for a while, then send a request. Arrival rate is users ÷ think time. Concurrency is arrival rate × time-in-system (Little's law).",
      "Memory sets how many sequences can be resident at once. Queueing theory sets what happens when arrivals exceed that: latency climbs long before throughput falls.",
      "Adding users is almost free while the batch is small, because one weight read serves the whole batch. Once KV traffic and the queue dominate, each extra user costs more than the last.",
    ],
    formula: "λ = users / think_time\nL = λ × W   (Little's law)",
    source: {
      label: "Little's law",
      href: "https://en.wikipedia.org/wiki/Little%27s_law",
    },
  },
  context: {
    title: "Context has two curves",
    kicker: "Context length",
    body: [
      "KV memory grows linearly with tokens. Prefill compute grows quadratically in the attention term, because every new token compares against every previous token.",
      "At 8k the MLP still dominates. By 128k, attention is most of the prefill FLOPs. That is why long-context requests hurt twice: they occupy more cache and they block the GPU longer.",
      "The practical result: doubling context roughly halves concurrency, and more than doubles time-to-first-token.",
    ],
    formula: "KV ∝ n\nprefill ≈ 2·P·n + 4·L·n²·d",
  },
  ttft: {
    title: "TTFT: time to first token",
    kicker: "Latency",
    body: [
      "TTFT is queue wait plus prefill. Prefill processes the whole prompt in parallel, so it is compute-bound and scales with prompt length.",
      "Caching attacks TTFT directly: an exact or semantic hit skips prefill entirely, and a prefix hit skips the shared part. This is why a cache hit feels instant.",
      "In this model TTFT is reported at p95 — the number your users actually complain about.",
    ],
    formula: "TTFT = queue_wait + prefill_time",
  },
  tpot: {
    title: "TPOT: time per output token",
    kicker: "Latency",
    body: [
      "Decode generates one token per sequence per step. Every step must read the weights and the whole active KV cache from HBM. It is bandwidth-bound, not compute-bound.",
      "Batching is nearly free on the weight read: 32 sequences share one pass over the weights. But each sequence adds KV bytes, so at long context the KV read starts to dominate.",
      "This is why a single request runs at 25 tok/s while the same GPU can emit 800 tok/s across a batch.",
    ],
    formula: "TPOT ≈ (weights + KV_active) / (HBM BW × efficiency)",
  },
  batching: {
    title: "Continuous batching",
    kicker: "Serving",
    body: [
      "Static batching waits for the slowest request in a batch. Continuous batching evicts finished sequences and admits new ones at every decode step, so the GPU stays full.",
      "This is the single biggest throughput lever in modern serving, and it is why 'users' and 'requests' are different units.",
      "The cost is tail latency: a long request shares the GPU with short ones, and queueing decides who waits.",
    ],
    source: {
      label: "Orca: continuous batching",
      href: "https://www.usenix.org/conference/osdi22/presentation/yu",
    },
  },
  exact: {
    title: "Exact response cache",
    kicker: "Cache tier 1",
    body: [
      "Hash the normalized prompt and replay the stored answer. Sub-millisecond, zero GPU work, and brittle: one character of difference is a miss.",
      "It shines for deterministic support answers, classification, extraction and template-generated content. It is dangerous for anything time-sensitive.",
      "A 12% hit rate is realistic for a consumer chatbot; for a fixed FAQ flow it can be far higher.",
    ],
  },
  semantic: {
    title: "Semantic cache",
    kicker: "Cache tier 2",
    body: [
      "Embed the prompt, find a near neighbour above a similarity threshold, and reuse that answer. It catches paraphrases exact matching misses.",
      "The threshold is a business decision: lower it and hit rate rises while wrong-answer risk rises with it. Never enable it for personalised or high-stakes answers without an eval set.",
      "Here the hit rate you set is taken as given; the point is how much latency and cost it removes.",
    ],
  },
  prefix: {
    title: "Prefix cache / KV reuse",
    kicker: "Cache tier 3",
    body: [
      "If many requests begin with the same tokens — a system prompt, tool schema, retrieved document, or the earlier turns of a conversation — their KV blocks are identical.",
      "A prefix cache stores those blocks once and lets every request point at them. PagedAttention makes this safe with block tables and copy-on-write; RadixAttention organises it as a tree.",
      "The win is both memory and time: shared tokens are stored once and their prefill is skipped. This is the highest-leverage cache in a real deployment.",
    ],
    formula: "saved = shared_prefix × hit_rate\nprefill ∝ 1 − saved  (TTFT also includes queue wait)",
    source: {
      label: "SGLang RadixAttention",
      href: "https://arxiv.org/abs/2312.07104",
    },
  },
  provider: {
    title: "Hosted prompt caching",
    kicker: "Cache tier 4",
    body: [
      "API providers sell the same idea: cache a long prefix server-side and bill cached input tokens at a fraction of the normal rate.",
      "Anthropic charges a write premium on first use, then cheap reads. OpenAI's automatic caching has no write premium and simply discounts repeated prefixes. Both pay off after a few reuses, which is why long system prompts and document Q&A benefit most.",
      "The multipliers here are editable and illustrative — check current provider pricing. The shape of the economics is what matters.",
    ],
    formula:
      "f = shared_prefix × hit_rate\nblended = (1−f) + f·read + f·(write−1)/reuses\nOpenAI: write = 1  ·  Anthropic: write ≈ 1.25, read ≈ 0.1",
  },
  quantization: {
    title: "Precision buys memory",
    kicker: "Optimisation",
    body: [
      "BF16 stores each weight in 2 bytes. FP8 halves that; INT4 quarters it. KV cache can be quantized separately, usually to FP8.",
      "Quantization is the cheapest way to make a model fit, but it is not free: quality degrades, and some kernels are faster than others. FP8 is the current serving default.",
      "Notice how halving precision roughly doubles the users you can hold — until the KV cache, not the weights, becomes the limit.",
    ],
    formula: "weights = params × bytes_per_param",
  },
  topology: {
    title: "When one GPU stops being enough",
    kicker: "Topology",
    body: [
      "There are five distinct reasons to add a GPU: the weights do not fit, the KV cache does not fit, decode is bandwidth-starved, prefill misses the TTFT target, or a single request's context exceeds one device.",
      "Each has a different fix. Sharding the model (tensor or pipeline parallelism) addresses the first and the last. More replicas address capacity and throughput. Neither fixes a bad interconnect.",
      "If the model already fits with room to spare, replicas are almost always better: no all-reduce, no cross-GPU dependency, linear capacity.",
    ],
  },
  tp: {
    title: "Tensor parallelism",
    kicker: "Topology",
    body: [
      "Tensor parallelism splits each layer's weight matrices across GPUs. Every GPU computes part of every layer, then the partial results are combined with an all-reduce.",
      "That all-reduce happens twice per layer, per token. On NVLink it is cheap; over PCIe it can dominate decode time. This is why TP is a within-node tool.",
      "Scaling is sublinear: communication, smaller per-GPU matrices, and synchronization overhead all take a cut. 8-way TP typically lands at 60–75% efficiency.",
    ],
    formula: "TPOT ∝ (weights/n + KV/n) / (n·BW) + all_reduce",
  },
  replicas: {
    title: "Replicas beat sharding when they fit",
    kicker: "Topology",
    body: [
      "A replica is a complete copy of the model on its own GPU. Requests are routed to whichever replica has room.",
      "No all-reduce, no cross-GPU synchronization, linear capacity, and a failure only takes out one replica. The catch: every replica duplicates the weights, so it only works when the model fits with room for KV.",
      "This simulator picks replicas when weights leave at least 15% of VRAM free, and shards otherwise.",
    ],
  },
  interconnect: {
    title: "The interconnect is the tax",
    kicker: "Topology",
    body: [
      "NVLink moves 900 GB/s or more between GPUs. PCIe 4.0 x16 moves about 64 GB/s. That is a 14× difference, and tensor parallelism depends on it.",
      "All-reduce traffic scales with batch size and hidden dimension, twice per layer. Over PCIe, the communication can cost more than the compute it is coordinating.",
      "Rule of thumb: shard inside an NVLink domain; replicate or pipeline across nodes.",
    ],
  },
  mfu: {
    title: "MFU and MBU",
    kicker: "Assumptions",
    body: [
      "Model FLOPs Utilisation is the fraction of peak compute actually achieved during prefill. Well-tuned serving kernels reach 35–55% on modern GPUs.",
      "Model Bandwidth Utilisation is the same idea for decode: the fraction of HBM bandwidth achieved. 60–75% is a healthy target.",
      "These are the two biggest sources of error in any capacity estimate. They are sliders here for exactly that reason.",
    ],
  },
  cost: {
    title: "Cost per million tokens",
    kicker: "Economics",
    body: [
      "Cost per token is GPU dollars per hour divided by tokens produced per hour. Throughput is the denominator, so batching and caching are the biggest cost levers.",
      "A tiny GPU can be more expensive per token than a large one if it is bandwidth-starved or under-batched. Utilisation matters more than sticker price.",
      "The output-token cost shown here excludes input processing; use it to compare configurations, not to set retail prices.",
    ],
    formula: "$/1M = (price_per_hour × GPUs / 3600) / tokens_per_sec × 1e6",
  },
  moe: {
    title: "Mixture of Experts",
    kicker: "Model shape",
    body: [
      "A sparse MoE stores every expert in memory but only routes each token through a couple of them. Mixtral 8×7B needs 46.7B parameters of VRAM but only does the FLOPs of 12.9B.",
      "That makes it memory-hungry and compute-cheap: you pay for capacity, then get dense-model latency at a fraction of the compute.",
      "In this simulator weights use total parameters, while prefill and decode FLOPs use active parameters.",
    ],
  },
  shared: {
    title: "Shared prefixes change the math",
    kicker: "Caching",
    body: [
      "When 35% of every request is an identical system prompt, that prefix is stored once instead of once per user. Memory drops, and prefill for those tokens is skipped.",
      "The more users share the prefix, the better the deal — which is why cache-aware routing (sending similar requests to the same replica) is a real optimisation.",
      "This also means more users can sometimes lower cost per token. Until capacity runs out.",
    ],
  },
};
