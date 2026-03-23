"""
Streamlit Frontend for LinkedIn Random Walker
=============================================
Run with: streamlit run app.py
"""

import streamlit as st
import numpy as np
import time
import os
import networkx as nx

# Add project root to path so we can import modules
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from graph.graph_builder import build_linkedin_graph
from graph.dataset import mask_labels, graph_statistics
from sampling.random_walker import RandomWalker
from sampling.gibbs_sampler import GibbsSampler
from propagation.label_propagation import LabelPropagation
from utils.metrics import compute_accuracy, compute_confusion_matrix

# Import visualizers
from utils.visualizer import (
    plot_graph,
    plot_convergence,
    plot_accuracy_comparison,
    plot_degree_distribution,
    plot_confusion_matrix,
)

st.set_page_config(
    page_title="LinkedIn Random Walker",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .reportview-container {
        background: #0D1117;
    }
    .main .block-container {
        padding-top: 2rem;
    }
    h1, h2, h3 {
        color: #e6edf3;
    }
</style>
""", unsafe_allow_html=True)

st.title("🔗 AI LinkedIn Random Walker")
st.markdown("**How AI guesses your job based on who you are connected to!**")

# ── Sidebar Configuration ───────────────────────────────────────────────
st.sidebar.header("⚙️ Simulation Settings")
st.sidebar.markdown("Change these numbers to see how they affect the AI's accuracy.")

num_nodes = st.sidebar.slider("Number of People (Nodes)", min_value=100, max_value=500, value=300, step=50)
known_fraction = st.sidebar.slider("Percent of Known Jobs", min_value=0.05, max_value=0.80, value=0.20, step=0.05)
p_in = st.sidebar.slider("Homophily (Likelihood to connect with SAME job)", min_value=0.10, max_value=0.80, value=0.30, step=0.05)
p_out = st.sidebar.slider("Likelihood to connect with DIFFERENT job", min_value=0.01, max_value=0.10, value=0.02, step=0.01)

st.sidebar.markdown("---")
gibbs_iters = st.sidebar.slider("AI 'Thinking' Time (Iterations)", min_value=50, max_value=300, value=200, step=50)

run_button = st.sidebar.button("🚀 Run Simulation", type="primary")

# ── Main Pipeline function ──────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def run_pipeline(n_nodes, k_frac, p_in_val, p_out_val, g_iters):
    import config
    # Set parameters for this run
    config.NUM_NODES = n_nodes
    config.KNOWN_LABEL_FRACTION = k_frac
    config.P_IN = p_in_val
    config.P_OUT = p_out_val
    config.GIBBS_ITERATIONS = g_iters
    
    # 1. Build Graph
    G, node_labels = build_linkedin_graph(num_nodes=n_nodes, p_in=p_in_val, p_out=p_out_val)
    stats = graph_statistics(G, node_labels)
    
    # 2. Mask Labels (hide them)
    observed_labels, known_nodes, unknown_nodes = mask_labels(G, node_labels, known_fraction=k_frac)
    
    # 3. Random Walker
    walker = RandomWalker(G, walk_length=20, num_walks=10) 
    walker.run_walks(observed_labels)
    
    # 4. Gibbs Sampler (The Core AI)
    sampler = GibbsSampler(
        G=G, observed_labels=observed_labels, unknown_nodes=unknown_nodes, 
        walker=walker, iterations=g_iters
    )
    gibbs_preds = sampler.run()
    
    # 5. Label Propagation (Mathematical Baseline)
    lp = LabelPropagation(G, observed_labels, unknown_nodes)
    lp_preds = lp.run()
    
    # 6. Evaluation metrics
    acc_gibbs = compute_accuracy(gibbs_preds, node_labels, unknown_nodes)
    acc_lp = compute_accuracy(lp_preds, node_labels, unknown_nodes)
    cm = compute_confusion_matrix(gibbs_preds, node_labels, unknown_nodes)
    
    # 7. Generate Visualisations
    # This will save images directly to the outputs/ folder
    pos = plot_graph(G, node_labels, "True Community Labels", "graph_true.png")
    plot_graph(G, {**observed_labels, **gibbs_preds}, "Gibbs Sampler Predicted Labels", "graph_gibbs.png", pos=pos)
    plot_convergence(sampler.convergence_curve)
    plot_confusion_matrix(cm)
    plot_degree_distribution(G)
    
    return {
        "stats": stats,
        "acc_gibbs": acc_gibbs,
        "acc_lp": acc_lp,
        "unknown_count": len(unknown_nodes),
    }

# ── UI Layout ───────────────────────────────────────────────────────────

if not run_button:
    st.info("👈 **Adjust the settings on the left, then click 'Run Simulation' to start!**")
    
    st.markdown("""
    ### Welcome to the LinkedIn AI Simulator! 👋
    
    **What is this?**
    Imagine a small professional network where people have jobs like *Software Engineer*, *Data Scientist*, or *Manager*. 
    In real life (**Homophily**), people usually connect with others who have the **same job** as them. 
    
    **The Challenge:**
    What if we **hide 80%** of the people's job titles? Can an AI figure out what everyone's job is, simply by looking at *who they are connected to*?
    
    **The AI Technique (Gibbs Sampling):**
    The AI looks at a person and says, *"Most of this person's connections are Data Scientists, so they are probably a Data Scientist too."* It does this for everyone, thousands of times, until it finds the most logical answer for the whole network!
    """)
else:
    with st.spinner("AI is analyzing the network... (this takes a few seconds)"):
        start_time = time.time()
        results = run_pipeline(num_nodes, known_fraction, p_in, p_out, gibbs_iters)
        elapsed = time.time() - start_time
        
    st.success(f"Simulation completed in {elapsed:.1f} seconds!")
    
    st.markdown("---")
    st.header("1. The Results 🏆")
    st.markdown("Here is how well the AI did at guessing the hidden job titles.")
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total People", results['stats']['num_nodes'])
    c2.metric("Connections", results['stats']['num_edges'])
    c3.metric("Hidden Jobs Guessed", results['unknown_count'])
    
    # The big reveal
    c4.metric(
        label="AI Prediction Accuracy", 
        value=f"{results['acc_gibbs']*100:.1f}%",
        delta="Gibbs Sampler"
    )
    
    st.markdown("---")
    
    # Graphs Section
    st.header("2. The Network Before & After 🕸️")
    st.markdown("""
    **Left:** The true network. Every color represents a different job (e.g. Blue = Software Engineer). 
    **Right:** The AI's guesses. Notice how closely it matches the real thing, even though it didn't know 80% of the colors!
    """)
    
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        st.image("outputs/graph_true.png", use_column_width=True)
    with col_g2:
        st.image("outputs/graph_gibbs.png", use_column_width=True)

    st.markdown("---")

    # Confusion matrix & Degree Distribution
    st.header("3. Digging Deeper 🔍")
    
    col_d1, col_d2 = st.columns(2)
    
    with col_d1:
        st.markdown('### Where Did the AI Make Mistakes?')
        st.markdown("""
        This **Confusion Matrix** shows what jobs the AI confused. 
        - The **diagonal dark blue boxes** are correct guesses.
        - Any numbers outside the diagonal are mistakes (e.g. guessing someone is a Manager when they are actually a Designer).
        """)
        st.image("outputs/confusion_matrix.png", use_column_width=True)
        
    with col_d2:
        st.markdown('### How Popular is Everyone?')
        st.markdown("""
        This **Degree Distribution** shows how many connections people have. 
        Most people have an average number of connections (the peak), while very few people are "super-connectors" with tons of friends.
        """)
        st.image("outputs/degree_distribution.png", use_column_width=True)

    st.markdown("---")
    
    st.header("4. How the AI 'Learned' over time 📈")
    st.markdown("""
    The **Convergence Curve** below shows the AI's "thought process". 
    At first (Iteration 0), it changes its mind about people's jobs constantly (high change rate). Over time, as it looks at more connections, it settles on the final answer and stops changing its mind. The AI has "converged"!
    """)
    st.image("outputs/convergence.png", use_column_width=True)

    st.success("🎉 **You're an AI Graph Expert!** Try changing the sliders on the left to make the game harder for the AI. (Hint: Lower the 'Homophily' slider to make the network random!)")
