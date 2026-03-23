# =============================================================================
# config.py – Global Configuration for LinkedIn Random Walker
# =============================================================================

# ── Graph Parameters ──────────────────────────────────────────────────────────
NUM_NODES       = 300          # Total number of nodes (LinkedIn users)
NUM_COMMUNITIES = 5            # Number of professional communities / labels
P_IN            = 0.30         # Intra-community edge probability  (homophily)
P_OUT           = 0.02         # Inter-community edge probability
RANDOM_SEED     = 42

# Professional communities (labels)
COMMUNITY_LABELS = [
    "Software Engineer",
    "Data Scientist",
    "Product Manager",
    "UX Designer",
    "Business Analyst",
]

# ── Semi-Supervised Setup ─────────────────────────────────────────────────────
KNOWN_LABEL_FRACTION = 0.20    # Fraction of nodes whose labels are revealed

# ── Random Walk Parameters ────────────────────────────────────────────────────
WALK_LENGTH      = 50          # Steps per random walk
NUM_WALKS        = 100         # Number of walks per node (for statistics)

# ── Gibbs Sampler Parameters ──────────────────────────────────────────────────
GIBBS_ITERATIONS = 200         # Total Gibbs sampling iterations
BURN_IN          = 50          # Burn-in period (discarded)
SMOOTHING        = 1.0         # Laplace smoothing for conditional probability

# ── Label Propagation Parameters ─────────────────────────────────────────────
LP_ITERATIONS    = 100         # Label propagation iterations
LP_ALPHA         = 0.85        # Propagation damping factor

# ── Output ────────────────────────────────────────────────────────────────────
OUTPUT_DIR       = "outputs"   # Directory to save figures
FIGURE_DPI       = 150
