# LinkedIn Random Walker – Project Explanation

## What is this project about?

Imagine **LinkedIn** as a map of people connected to each other. This project answers:

> **"If you only know the job title of *some* people, can you guess the job title of the rest — just by looking at who they're connected to?"**

---

## The 3 Big Ideas

### 1. The Graph (LinkedIn Network)
- Every **node** = a LinkedIn user
- Every **edge** = a connection between two users
- Each user has a **profession** (label): Software Engineer, Data Scientist, Product Manager, UX Designer, Business Analyst

### 2. Homophily — *"birds of a feather flock together"*
- People tend to connect with others **in the same profession**
- A Software Engineer is more likely to connect with other Software Engineers than with UX Designers
- This is the key property the algorithm exploits

### 3. Gibbs Sampling — the AI part
- We **hide 80% of the labels** (pretend we don't know most people's jobs)
- Gibbs Sampling **guesses each person's job** by looking at their neighbors:
  > *"Most of my connections are Data Scientists → I'm probably a Data Scientist too"*
- It repeats this thousands of times, and the **most frequent guess wins**
- This is a type of **MCMC (Markov Chain Monte Carlo)** — a probabilistic AI technique

---

## The 9-Step Pipeline

```
Build Graph → Compute Stats → Hide 80% of Labels
     → Random Walk → Gibbs Sampling → Label Propagation
          → Evaluate → Plot Results → Print Report
```

| Step | What happens |
|---|---|
| **Build Graph** | Creates 300 fake LinkedIn users in 5 professions |
| **Homophily check** | Verifies that 78% of connections are within the same profession |
| **Hide labels** | Hides job titles for 240/300 people |
| **Random Walker** | "Walks" through the network to learn neighborhood patterns |
| **Gibbs Sampler** | Guesses the 240 hidden jobs using MCMC |
| **Label Propagation** | A simpler baseline method that also tries to guess the labels |
| **Evaluation** | Checks how accurate the guesses were |
| **Visualizations** | Saves 8 charts (graph plots, accuracy bar charts, etc.) |

---

## Results

| Method | Accuracy |
|---|---|
| **Gibbs Sampling** (core AI) | **90.83%** |
| Label Propagation (baseline) | 100.00% |

The Gibbs Sampler correctly guessed the profession of **218 out of 240 hidden people**, just from looking at who they're connected to!

---

## In One Sentence

> **This project simulates a LinkedIn network and uses a probabilistic AI algorithm (Gibbs Sampling) to figure out people's job titles by looking at who they're friends with — because similar professionals tend to connect.**
