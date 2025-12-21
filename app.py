#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
WEB INTERFACE FOR COMPLETE HIERARCHICAL ANALYSIS
Streamlit interface for hierarchical analysis of generated molecules
Academic version - Clean design - Improved
"""

import streamlit as st
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from dataclasses import dataclass
import os
import json
from tqdm import tqdm
import numpy as np
from collections import Counter
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, DataStructs, Crippen, Lipinski
from rdkit import rdBase
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import tempfile
import base64
import pickle

# Désactiver les logs RDKit
rdBase.DisableLog('rdApp.error')

# --- CONFIGURATION ---
DRIVE_PATH = '.'
VOCAB_FILE = os.path.join(DRIVE_PATH, 'vocab_dataset.json')
CHECKPOINT_FILE = os.path.join(DRIVE_PATH, '.', 'cond_gpt_categorical_extended.pth')
DATA_FILE = 's_100_str_+1M_fixed.txt'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- FONCTIONS IDENTIQUES (adaptées pour Streamlit) ---
def has_functional_group(mol):
    smarts_patterns = [
        '[OH]',
        '[#6]C(=O)[O;H0]',
        'C(=O)[OH]',
        '[NH2]'
    ]
    for pattern in smarts_patterns:
        if mol.HasSubstructMatch(Chem.MolFromSmarts(pattern)):
            return 1.0
    return 0.0

def calculate_r_value(mol):
    try:
        mol_wt = Descriptors.MolWt(mol)
        logp = Crippen.MolLogP(mol)
        if mol_wt > 0:
            r_value = logp / (mol_wt / 100)
            return r_value
        else:
            return 0.0
    except:
        return 0.0

# --- DÉFINITION DES CONDITIONS ---
CONDITIONS = {
    0: {
        "name": "Condition 1: LogP ≤ 3",
        "description": "Single objective: logP ≤ 3",
        "get_vector": lambda mol: [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    },
    1: {
        "name": "Condition 2: Structural",
        "description": "2 aromatic rings, 1 non-aromatic, functional groups, R-value [0.05-0.50]",
        "get_vector": lambda mol: [0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 1.0, 1.0, 1.0, 0.0]
    },
    2: {
        "name": "Condition 3: Lipinski Ro3",
        "description": "LogP≤3, MW≤480, HBA≤3, HBD≤3, RotB≤3",
        "get_vector": lambda mol: [1.0, 1.0, 3.0, 3.0, 3.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    },
    3: {
        "name": "Condition 4: Structural + Lipinski",
        "description": "Combination of conditions 2 and 3",
        "get_vector": lambda mol: [1.0, 1.0, 3.0, 3.0, 3.0, 2.0, 1.0, 1.0, 1.0, 1.0]
    }
}

def evaluate_condition(mol, condition_idx):
    """Évalue si une molécule satisfait une condition spécifique"""
    try:
        if condition_idx == 0:  # LogP ≤ 3
            logp = Crippen.MolLogP(mol)
            return logp <= 3.0

        elif condition_idx == 1:  # Structural Objectives
            aromatic_rings = Lipinski.NumAromaticRings(mol)
            non_aromatic_rings = Lipinski.NumAliphaticRings(mol)
            ring_condition = (aromatic_rings == 2) and (non_aromatic_rings == 1)
            functional_group_condition = has_functional_group(mol)
            r_value = calculate_r_value(mol)
            r_value_condition = (0.05 <= r_value <= 0.50)
            return ring_condition and functional_group_condition and r_value_condition

        elif condition_idx == 2:  # Lipinski's Rule of Three
            logp = Crippen.MolLogP(mol)
            mw = Descriptors.MolWt(mol)
            hbd = Lipinski.NumHDonors(mol)
            hba = Lipinski.NumHAcceptors(mol)
            rotb = Lipinski.NumRotatableBonds(mol)
            return (logp <= 3.0) and (mw <= 480) and (hbd <= 3) and (hba <= 3) and (rotb <= 3)

        elif condition_idx == 3:  # Structural + Lipinski
            aromatic_rings = Lipinski.NumAromaticRings(mol)
            non_aromatic_rings = Lipinski.NumAliphaticRings(mol)
            ring_condition = (aromatic_rings == 2) and (non_aromatic_rings == 1)
            functional_group_condition = has_functional_group(mol)
            r_value = calculate_r_value(mol)
            r_value_condition = (0.05 <= r_value <= 0.50)
            logp = Crippen.MolLogP(mol)
            mw = Descriptors.MolWt(mol)
            hbd = Lipinski.NumHDonors(mol)
            hba = Lipinski.NumHAcceptors(mol)
            rotb = Lipinski.NumRotatableBonds(mol)
            lipinski_condition = (logp <= 3.0) and (mw <= 480) and (hbd <= 3) and (hba <= 3) and (rotb <= 3)
            return ring_condition and functional_group_condition and r_value_condition and lipinski_condition

        else:
            return False
    except:
        return False

# --- ARCHITECTURE MODÈLE ---
@dataclass
class GPTConfig:
    block_size: int = 128
    vocab_size: int = 57
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.1
    condition_dim: int = 10

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class ConditionalDrugGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.condition_projector = nn.Sequential(
            nn.Linear(config.condition_dim, config.n_embd),
            nn.ReLU()
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, conditions=None):
        device = idx.device
        B, T = idx.shape
        assert T <= self.config.block_size, f"Séquence trop longue: {T}"
        pos = torch.arange(0, T, dtype=torch.long, device=device).unsqueeze(0)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        assert conditions is not None, "Les conditions doivent être fournies !"
        cond_emb = self.condition_projector(conditions)
        x = self.transformer.drop(tok_emb + pos_emb + cond_emb.unsqueeze(1))
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None
        return logits, loss

# --- FONCTIONS D'ANALYSE ADAPTÉES POUR STREAMLIT ---
@st.cache_data
def load_reference_smiles(reference_smiles_file):
    """Charge le dataset de référence avec cache"""
    reference_smiles = set()
    if os.path.exists(reference_smiles_file):
        with open(reference_smiles_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    mol = Chem.MolFromSmiles(line)
                    if mol:
                        canon_smiles = Chem.MolToSmiles(mol)
                        reference_smiles.add(canon_smiles)
    return reference_smiles

@st.cache_resource
def load_model_and_vocab():
    """Charge le modèle et le vocabulaire avec cache"""
    with open(VOCAB_FILE, 'r', encoding='utf-8') as f:
        vocab_data = json.load(f)
    stoi = vocab_data['stoi']
    itos = vocab_data['itos']
    
    config = GPTConfig(vocab_size=len(stoi))
    model = ConditionalDrugGPT(config)
    model.to(DEVICE)
    
    try:
        checkpoint = torch.load(CHECKPOINT_FILE, map_location=DEVICE, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
    except:
        checkpoint = torch.load(CHECKPOINT_FILE, map_location=DEVICE, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval()
    return model, stoi, itos

def comprehensive_hierarchical_analysis(generated_smiles, reference_smiles_file, progress_bar=None, status_text=None):
    """Analyse hiérarchique complète avec barres de progression pour Streamlit"""
    if status_text:
        status_text.text("Démarrage de l'analyse hiérarchique...")
    
    results = {
        'total_generated': len(generated_smiles),
        'total_valid': 0,
        'total_novel': 0,
        'total_unique_novel': 0,
        'validity_percentage': 0.0,
        'novelty_percentage': 0.0,
        'uniqueness_percentage': 0.0,
        'condition_results': {},
        'unique_novel_molecules': {},
        'unique_novel_smiles': []
    }
    
    # Charger le dataset de référence
    if status_text:
        status_text.text("Chargement du dataset de référence...")
    
    reference_smiles = load_reference_smiles(reference_smiles_file)
    
    # Niveau 1: Validation
    if status_text:
        status_text.text("Niveau 1: Validation des SMILES...")
    
    valid_molecules = []
    valid_smiles = []
    
    validation_progress = st.progress(0, text="Validation des molécules générées...")
    for i, smiles in enumerate(generated_smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            valid_molecules.append(mol)
            valid_smiles.append(smiles)
        
        if (i + 1) % max(1, len(generated_smiles) // 100) == 0:
            validation_progress.progress((i + 1) / len(generated_smiles))
    
    validation_progress.empty()
    
    results['total_valid'] = len(valid_molecules)
    if results['total_generated'] > 0:
        results['validity_percentage'] = (results['total_valid'] / results['total_generated']) * 100
    
    # Niveau 2: Novelty
    if status_text:
        status_text.text("Niveau 2: Vérification de la nouveauté...")
    
    novel_molecules = []
    novel_smiles = []
    
    novelty_progress = st.progress(0, text="Vérification de la nouveauté...")
    for i, (mol, smiles) in enumerate(zip(valid_molecules, valid_smiles)):
        canon_smiles = Chem.MolToSmiles(mol)
        if canon_smiles not in reference_smiles:
            novel_molecules.append(mol)
            novel_smiles.append(smiles)
        
        if (i + 1) % max(1, len(valid_molecules) // 100) == 0:
            novelty_progress.progress((i + 1) / len(valid_molecules))
    
    novelty_progress.empty()
    
    results['total_novel'] = len(novel_molecules)
    if results['total_valid'] > 0:
        results['novelty_percentage'] = (results['total_novel'] / results['total_valid']) * 100
    
    # Niveau 3: Uniqueness parmi les Novel
    if status_text:
        status_text.text("Niveau 3: Vérification de l'unicité...")
    
    unique_novel_molecules = {}
    unique_novel_smiles = []
    
    uniqueness_progress = st.progress(0, text="Déduplication des molécules nouvelles...")
    for i, (mol, smiles) in enumerate(zip(novel_molecules, novel_smiles)):
        canon_smiles = Chem.MolToSmiles(mol)
        if canon_smiles not in unique_novel_molecules:
            unique_novel_molecules[canon_smiles] = mol
            unique_novel_smiles.append(smiles)
        
        if (i + 1) % max(1, len(novel_molecules) // 100) == 0:
            uniqueness_progress.progress((i + 1) / len(novel_molecules))
    
    uniqueness_progress.empty()
    
    results['total_unique_novel'] = len(unique_novel_molecules)
    results['unique_novel_molecules'] = unique_novel_molecules
    results['unique_novel_smiles'] = unique_novel_smiles
    
    if results['total_novel'] > 0:
        results['uniqueness_percentage'] = (results['total_unique_novel'] / results['total_novel']) * 100
    
    # Niveau 4: Satisfaction des conditions
    if status_text:
        status_text.text("Niveau 4: Évaluation des conditions...")
    
    condition_results = {
        'total_unique_novel': results['total_unique_novel'],
        'condition_counts': {0: 0, 1: 0, 2: 0, 3: 0},
        'condition_percentages': {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0},
        'examples_per_condition': {0: [], 1: [], 2: [], 3: []},
        'properties_per_condition': {0: [], 1: [], 2: [], 3: []}
    }
    
    for condition_idx in range(4):
        condition_progress = st.progress(0, text=f"Évaluation de la condition {condition_idx+1}...")
        condition_count = 0
        
        for i, (canon_smiles, mol) in enumerate(unique_novel_molecules.items()):
            if evaluate_condition(mol, condition_idx):
                condition_count += 1
                
                # Garder des exemples
                if len(condition_results['examples_per_condition'][condition_idx]) < 5:
                    condition_results['examples_per_condition'][condition_idx].append(canon_smiles)
                    
                    # Propriétés
                    props = {
                        'SMILES': canon_smiles,
                        'LogP': Crippen.MolLogP(mol),
                        'MW': Descriptors.MolWt(mol),
                        'HBD': Lipinski.NumHDonors(mol),
                        'HBA': Lipinski.NumHAcceptors(mol),
                        'RotB': Lipinski.NumRotatableBonds(mol),
                    }
                    if condition_idx in [1, 3]:  # Conditions structurelles
                        props.update({
                            'Aromatic': Lipinski.NumAromaticRings(mol),
                            'Aliphatic': Lipinski.NumAliphaticRings(mol),
                            'R-value': calculate_r_value(mol),
                            'Functional': "Oui" if has_functional_group(mol) else "Non"
                        })
                    condition_results['properties_per_condition'][condition_idx].append(props)
            
            if (i + 1) % max(1, len(unique_novel_molecules) // 100) == 0:
                condition_progress.progress((i + 1) / len(unique_novel_molecules))
        
        condition_progress.empty()
        
        condition_results['condition_counts'][condition_idx] = condition_count
        if results['total_unique_novel'] > 0:
            condition_results['condition_percentages'][condition_idx] = (condition_count / results['total_unique_novel']) * 100
    
    results['condition_results'] = condition_results
    
    if status_text:
        status_text.text("Analyse hiérarchique terminée !")
    
    return results

@torch.no_grad()
def generate_molecules(model, condition_tensor, stoi, itos, start_idx, end_idx, 
                      num_molecules=10000, temperature=0.6, progress_bar=None, status_text=None):
    """Génère des molécules avec température spécifique et progression pour Streamlit"""
    if status_text:
        status_text.text(f"Génération de {num_molecules} molécules (température: {temperature})...")
    
    generated_smiles = []
    
    generation_progress = st.progress(0, text=f"Génération des molécules...")
    
    with torch.no_grad():
        for i in range(num_molecules):
            top_k = 30
            idx = torch.tensor([[start_idx]], dtype=torch.long, device=DEVICE)
            condition_local = condition_tensor.to(DEVICE)

            for step in range(80):
                idx_cond = idx if idx.size(1) <= 128 else idx[:, -128:]
                logits, _ = model(idx_cond, conditions=condition_local)
                logits = logits[:, -1, :] / temperature

                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

                probs = F.softmax(logits, dim=-1)
                probs_mod = probs.clone()
                probs_mod[0, start_idx] = 0.0
                if step < 8:
                    probs_mod[0, end_idx] = 0.0

                if probs_mod.sum() > 0:
                    probs_mod = probs_mod / probs_mod.sum()
                else:
                    probs_mod = probs

                idx_next = torch.multinomial(probs_mod, num_samples=1)
                next_token = idx_next.item()

                if next_token == end_idx and step >= 8:
                    break

                idx = torch.cat((idx, idx_next), dim=1)

            tokens = idx[0].tolist()
            if len(tokens) > 1:
                tokens_to_decode = tokens[1:]
                if end_idx in tokens_to_decode:
                    end_pos = tokens_to_decode.index(end_idx)
                    tokens_to_decode = tokens_to_decode[:end_pos]
            else:
                tokens_to_decode = []

            smiles = ''.join([itos[str(i)] for i in tokens_to_decode if str(i) in itos])

            if smiles:
                generated_smiles.append(smiles)
            
            # Mettre à jour la barre de progression
            if (i + 1) % max(1, num_molecules // 100) == 0:
                generation_progress.progress((i + 1) / num_molecules)
    
    generation_progress.empty()
    
    if status_text:
        status_text.text(f"{len(generated_smiles)} molécules générées avec succès !")
    
    return generated_smiles

def calculate_intdiv(smiles_list, status_text=None):
    """Calcule la diversité interne sur un échantillon avec optimisation"""
    if status_text:
        status_text.text("Calcul de la diversité interne (IntDiv)...")
    
    if len(smiles_list) < 2:
        return 0.0
    
    # Échantillonnage pour performance
    sample_size = min(2000, len(smiles_list))  # Augmenté à 2000 pour plus de précision
    if len(smiles_list) > sample_size:
        indices = np.random.choice(len(smiles_list), sample_size, replace=False)
        sample_smiles = [smiles_list[i] for i in indices]
    else:
        sample_smiles = smiles_list
    
    fingerprints = []
    
    fingerprint_progress = st.progress(0, text="Calcul des empreintes moléculaires...")
    for i, smiles in enumerate(sample_smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
            fingerprints.append(fp)
        
        if (i + 1) % max(1, len(sample_smiles) // 100) == 0:
            fingerprint_progress.progress((i + 1) / len(sample_smiles))
    
    fingerprint_progress.empty()
    
    if len(fingerprints) < 2:
        return 0.0
    
    # Utilisation de BulkTanimotoSimilarity pour optimisation
    similarities = []
    similarity_progress = st.progress(0, text="Calcul des similarités...")
    
    # Calculer les similarités par paires de manière optimisée
    for i in range(len(fingerprints)):
        sims = DataStructs.BulkTanimotoSimilarity(fingerprints[i], fingerprints[i+1:])
        similarities.extend(sims)
        
        if (i + 1) % max(1, len(fingerprints) // 100) == 0:
            similarity_progress.progress((i + 1) / len(fingerprints))
    
    similarity_progress.empty()
    
    if similarities:
        mean_similarity = np.mean(similarities)
        intdiv = 1 - mean_similarity
        return intdiv
    else:
        return 0.0

def create_visualization(results):
    """Crée des visualisations pour les résultats"""
    
    # Données pour les graphiques
    categories = ['Générées', 'Valides', 'Nouvelles', 'Uniques']
    counts = [
        results['total_generated'],
        results['total_valid'],
        results['total_novel'],
        results['total_unique_novel']
    ]
    
    percentages = [
        100.0,  # Générées (référence)
        results['validity_percentage'],
        results['novelty_percentage'],
        results['uniqueness_percentage']
    ]
    
    # Graphique 1: Hiérarchie de qualité
    fig1 = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Nombre de molécules', 'Pourcentages relatifs'),
        specs=[[{'type': 'bar'}, {'type': 'bar'}]]
    )
    
    fig1.add_trace(
        go.Bar(x=categories, y=counts, name='Nombre', marker_color='#2C3E50'),
        row=1, col=1
    )
    
    fig1.add_trace(
        go.Bar(x=categories[1:], y=percentages[1:], name='Pourcentage', marker_color='#3498DB'),
        row=1, col=2
    )
    
    fig1.update_layout(
        title='Hiérarchie de qualité des molécules',
        showlegend=False,
        height=400,
        font=dict(family="Arial, sans-serif", size=12),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    # Graphique 2: Satisfaction des conditions
    condition_names = [CONDITIONS[i]["name"] for i in range(4)]
    condition_counts = [results['condition_results']['condition_counts'][i] for i in range(4)]
    condition_percentages = [results['condition_results']['condition_percentages'][i] for i in range(4)]
    
    fig2 = go.Figure(data=[
        go.Bar(name='Nombre', x=condition_names, y=condition_counts, marker_color='#2C3E50'),
        go.Bar(name='Pourcentage', x=condition_names, y=condition_percentages, marker_color='#3498DB', 
               yaxis='y2')
    ])
    
    fig2.update_layout(
        title='Satisfaction des conditions',
        yaxis=dict(title='Nombre de molécules'),
        yaxis2=dict(title='Pourcentage (%)', overlaying='y', side='right'),
        barmode='group',
        height=400,
        font=dict(family="Arial, sans-serif", size=12),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    return fig1, fig2

def save_results_to_file(results, condition_option, temperature, num_molecules):
    """Sauvegarde les résultats dans un fichier pickle"""
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    filename = f"results_{timestamp}.pkl"
    
    data_to_save = {
        'results': results,
        'parameters': {
            'condition': condition_option,
            'temperature': temperature,
            'num_molecules': num_molecules,
            'timestamp': timestamp
        }
    }
    
    with open(filename, 'wb') as f:
        pickle.dump(data_to_save, f)
    
    return filename

def load_results_from_file(uploaded_file):
    """Charge les résultats depuis un fichier pickle"""
    try:
        data = pickle.load(uploaded_file)
        return data['results'], data['parameters']
    except Exception as e:
        st.error(f"Erreur lors du chargement: {e}")
        return None, None

# --- INTERFACE STREAMLIT ---
def main():
    st.set_page_config(
        page_title="Analyse Hiérarchique de Molécules Générées",
        page_icon="⚗️",
        layout="wide"
    )
    
    # CSS personnalisé - Design académique
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.2rem;
        color: #2C3E50;
        text-align: center;
        margin-bottom: 1.5rem;
        font-weight: 600;
        border-bottom: 2px solid #3498DB;
        padding-bottom: 0.5rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #2C3E50;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        font-weight: 500;
        border-left: 4px solid #3498DB;
        padding-left: 0.5rem;
    }
    .subsection-header {
        font-size: 1.2rem;
        color: #34495E;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
        font-weight: 500;
    }
    .metric-card {
        background-color: #FFFFFF;
        border-radius: 8px;
        padding: 1.2rem;
        margin: 0.5rem;
        box-shadow: 0 2px 6px rgba(0,0,0,0.08);
        border-left: 4px solid #3498DB;
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.12);
    }
    .metric-value {
        font-size: 2rem;
        color: #2C3E50;
        font-weight: 600;
        margin: 0.5rem 0;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #7F8C8D;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .data-table {
        background-color: #FFFFFF;
        border-radius: 6px;
        padding: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .info-box {
        background-color: #F8F9FA;
        border-left: 4px solid #3498DB;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 4px;
    }
    .success-box {
        background-color: #E8F5E9;
        border-left: 4px solid #2ECC71;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 4px;
    }
    .warning-box {
        background-color: #FFF3E0;
        border-left: 4px solid #F39C12;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 4px;
    }
    .button-primary {
        background-color: #3498DB !important;
        color: white !important;
        border: none !important;
        padding: 0.5rem 1.5rem !important;
        border-radius: 4px !important;
        font-weight: 500 !important;
    }
    .button-secondary {
        background-color: #2C3E50 !important;
        color: white !important;
        border: none !important;
        padding: 0.5rem 1.5rem !important;
        border-radius: 4px !important;
        font-weight: 500 !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # En-tête principal
    st.markdown('<h1 class="main-header">Analyse Hiérarchique de Molécules Générées</h1>', unsafe_allow_html=True)
    
    # Description
    with st.expander("Description de l'analyse hiérarchique", expanded=True):
        st.markdown("""
        **Hiérarchie d'analyse**:
        1. **Molécules Générées** → 2. **Valides** → 3. **Nouvelles** → 4. **Uniques Nouvelles** → 5. **Conditions**
        
        Cette analyse suit une approche hiérarchique stricte où chaque niveau dépend du précédent. 
        Seules les molécules passant tous les filtres sont retenues pour l'analyse finale.
        """)
    
    # Sidebar pour les paramètres
    with st.sidebar:
        st.markdown('<div class="section-header">Paramètres de Génération</div>', unsafe_allow_html=True)
        
        # Sélection de la condition
        condition_option = st.selectbox(
            "Condition de génération",
            options=list(CONDITIONS.keys()),
            format_func=lambda x: CONDITIONS[x]["name"],
            help=CONDITIONS[2]["description"] if len(CONDITIONS) > 2 else ""
        )
        
        # Affichage de la description de la condition
        if condition_option in CONDITIONS:
            st.markdown(f'<div class="info-box"><strong>{CONDITIONS[condition_option]["name"]}:</strong> {CONDITIONS[condition_option]["description"]}</div>', unsafe_allow_html=True)
        
        # Nombre de molécules
        num_molecules = st.slider(
            "Nombre de molécules à générer",
            min_value=100,
            max_value=50000,
            value=10000,
            step=100,
            help="Nombre total de molécules à générer"
        )
        
        # Température
        temperature = st.slider(
            "Température d'échantillonnage",
            min_value=0.1,
            max_value=2.0,
            value=0.6,
            step=0.1,
            help="Contrôle la créativité de la génération (plus élevé = plus diversifié)"
        )
        
        # Options avancées
        with st.expander("Options d'analyse avancées"):
            calculate_intdiv_option = st.checkbox(
                "Calculer la diversité interne (IntDiv)",
                value=True,
                help="Calcule la diversité des molécules uniques et nouvelles"
            )
            
            show_examples = st.checkbox(
                "Afficher des exemples par condition",
                value=True,
                help="Affiche quelques exemples de molécules satisfaisant chaque condition"
            )
        
        # Bouton de lancement
        st.markdown("---")
        generate_button = st.button(
            "Lancer la génération et l'analyse",
            type="primary",
            use_container_width=True
        )
        
        # Section sauvegarde/chargement
        st.markdown('<div class="section-header">Sauvegarde et Chargement</div>', unsafe_allow_html=True)
        
        save_button = st.button(
            "Sauvegarder les résultats actuels",
            disabled=('results' not in st.session_state or st.session_state.results is None),
            use_container_width=True
        )
        
        uploaded_file = st.file_uploader(
            "Charger des résultats précédents (.pkl)",
            type=['pkl'],
            help="Chargez un fichier de résultats sauvegardé"
        )
        
        if uploaded_file is not None:
            load_button = st.button(
                "Charger les résultats",
                use_container_width=True
            )
    
    # Contenu principal
    if generate_button:
        # Initialisation de la session
        if 'results' not in st.session_state:
            st.session_state.results = None
        
        # Conteneurs pour les résultats
        progress_container = st.container()
        results_container = st.container()
        
        with progress_container:
            status_text = st.empty()
            try:
                # Chargements avec cache
                status_text.text("Chargement du vocabulaire et du modèle...")
                model, stoi, itos = load_model_and_vocab()
                start_token = stoi['<start>']
                end_token = stoi['<end>']
                
                # Condition de génération
                condition_vector = CONDITIONS[condition_option]["get_vector"](None)
                condition_tensor = torch.tensor([condition_vector], dtype=torch.float32)
                
                # Génération
                generated_smiles = generate_molecules(
                    model, condition_tensor, stoi, itos, start_token, end_token,
                    num_molecules=num_molecules, temperature=temperature,
                    status_text=status_text
                )
                
                # Analyse hiérarchique
                results = comprehensive_hierarchical_analysis(
                    generated_smiles, DATA_FILE, status_text=status_text
                )
                
                # Calcul de la diversité si demandé
                if calculate_intdiv_option and results['total_unique_novel'] > 0:
                    unique_novel_smiles = results['unique_novel_smiles']
                    intdiv = calculate_intdiv(unique_novel_smiles, status_text)
                    results['intdiv'] = intdiv
                else:
                    results['intdiv'] = None
                
                # Stockage des résultats
                st.session_state.results = results
                status_text.text("Analyse terminée avec succès !")
                
            except Exception as e:
                status_text.text(f"Erreur: {str(e)}")
                st.error(f"Une erreur s'est produite: {str(e)}")
    
    # Gestion de la sauvegarde
    if save_button and 'results' in st.session_state and st.session_state.results is not None:
        try:
            filename = save_results_to_file(
                st.session_state.results, 
                condition_option if 'condition_option' in locals() else 2,
                temperature if 'temperature' in locals() else 0.6,
                num_molecules if 'num_molecules' in locals() else 10000
            )
            st.success(f"Résultats sauvegardés dans {filename}")
        except Exception as e:
            st.error(f"Erreur lors de la sauvegarde: {e}")
    
    # Gestion du chargement
    if uploaded_file is not None and load_button:
        results, params = load_results_from_file(uploaded_file)
        if results is not None:
            st.session_state.results = results
            st.success("Résultats chargés avec succès !")
            if params:
                st.info(f"Paramètres: Condition {params.get('condition', 'N/A')}, Température {params.get('temperature', 'N/A')}, {params.get('num_molecules', 'N/A')} molécules")
    
    # Affichage des résultats
    if 'results' in st.session_state and st.session_state.results:
        results = st.session_state.results
        
        with results_container:
            st.markdown('<div class="section-header">Résultats de l\'Analyse</div>', unsafe_allow_html=True)
            
            # Métriques principales
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Molécules Générées</div>
                    <div class="metric-value">{results['total_generated']:,}</div>
                    <div>Total initial</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Molécules Valides</div>
                    <div class="metric-value">{results['total_valid']:,}</div>
                    <div>{results['validity_percentage']:.1f}% du total</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Molécules Nouvelles</div>
                    <div class="metric-value">{results['total_novel']:,}</div>
                    <div>{results['novelty_percentage']:.1f}% des valides</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Uniques Nouvelles</div>
                    <div class="metric-value">{results['total_unique_novel']:,}</div>
                    <div>{results['uniqueness_percentage']:.1f}% des nouvelles</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Visualisations
            st.markdown('<div class="section-header">Visualisations des Résultats</div>', unsafe_allow_html=True)
            
            fig1, fig2 = create_visualization(results)
            
            col_viz1, col_viz2 = st.columns(2)
            with col_viz1:
                st.plotly_chart(fig1, use_container_width=True)
            with col_viz2:
                st.plotly_chart(fig2, use_container_width=True)
            
            # Diversité interne
            if results.get('intdiv') is not None:
                st.markdown('<div class="section-header">Analyse de Diversité</div>', unsafe_allow_html=True)
                
                intdiv_value = results['intdiv']
                
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Diversité Interne (IntDiv)</div>
                    <div class="metric-value">{intdiv_value:.4f}</div>
                    <div>Calculée sur {min(2000, results['total_unique_novel'])} molécules uniques nouvelles</div>
                    <div><small>IntDiv = 1 - similarité moyenne (1 = diversité maximale)</small></div>
                </div>
                """, unsafe_allow_html=True)
            
            # Détails par condition
            st.markdown('<div class="section-header">Analyse par Condition</div>', unsafe_allow_html=True)
            
            for condition_idx in range(4):
                with st.expander(f"{CONDITIONS[condition_idx]['name']}", expanded=(condition_idx==0)):
                    count = results['condition_results']['condition_counts'][condition_idx]
                    percentage = results['condition_results']['condition_percentages'][condition_idx]
                    
                    st.markdown(f"""
                    <div class="info-box">
                        <strong>Description:</strong> {CONDITIONS[condition_idx]['description']}<br>
                        <strong>Molécules satisfaisantes:</strong> {count}/{results['total_unique_novel']} ({percentage:.1f}%)
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Exemples
                    if show_examples and results['condition_results']['examples_per_condition'][condition_idx]:
                        st.markdown('<div class="subsection-header">Exemples de molécules</div>', unsafe_allow_html=True)
                        for i, smiles in enumerate(results['condition_results']['examples_per_condition'][condition_idx][:3]):
                            col_sm, col_prop = st.columns([2, 3])
                            with col_sm:
                                st.code(smiles, language='text')
                            
                            # Propriétés
                            if i < len(results['condition_results']['properties_per_condition'][condition_idx]):
                                with col_prop:
                                    props = results['condition_results']['properties_per_condition'][condition_idx][i]
                                    df_props = pd.DataFrame([props])
                                    st.dataframe(df_props, use_container_width=True, hide_index=True)
            
            # Téléchargement des résultats
            st.markdown('<div class="section-header">Export des Données</div>', unsafe_allow_html=True)
            
            # Préparation des données pour export
            export_data = {
                'parametres': {
                    'condition': CONDITIONS[condition_option if 'condition_option' in locals() else 2]["name"],
                    'temperature': temperature if 'temperature' in locals() else 0.6,
                    'nombre_molecules': num_molecules if 'num_molecules' in locals() else 10000
                },
                'resultats_generaux': {
                    'total_generes': results['total_generated'],
                    'total_valides': results['total_valid'],
                    'pourcentage_valides': results['validity_percentage'],
                    'total_nouvelles': results['total_novel'],
                    'pourcentage_nouvelles': results['novelty_percentage'],
                    'total_uniques_nouvelles': results['total_unique_novel'],
                    'pourcentage_uniques': results['uniqueness_percentage'],
                    'diversite_interne': results.get('intdiv', 'Non calculée')
                },
                'resultats_conditions': {}
            }
            
            for cond_idx in range(4):
                export_data['resultats_conditions'][cond_idx] = {
                    'nom': CONDITIONS[cond_idx]['name'],
                    'satisfaites': results['condition_results']['condition_counts'][cond_idx],
                    'pourcentage': results['condition_results']['condition_percentages'][cond_idx]
                }
            
            # Export JSON
            json_str = json.dumps(export_data, indent=2, ensure_ascii=False)
            st.download_button(
                label="Télécharger les résultats (JSON)",
                data=json_str,
                file_name=f"analyse_molecules_{time.strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )
            
            # Export des molécules uniques
            if results['unique_novel_smiles']:
                smiles_text = '\n'.join(results['unique_novel_smiles'])
                st.download_button(
                    label="Télécharger les SMILES uniques",
                    data=smiles_text,
                    file_name=f"smiles_uniques_{time.strftime('%Y%m%d_%H%M%S')}.smi",
                    mime="text/plain",
                    use_container_width=True
                )
    
    else:
        # Page d'accueil
        st.markdown("""
        <div class="info-box">
            <h3>Interface d'Analyse Hiérarchique de Molécules Générées</h3>
            <p>Cette application permet de réaliser une analyse hiérarchique complète de molécules générées par un modèle GPT conditionnel.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="section-header">Méthodologie d\'Analyse</div>', unsafe_allow_html=True)
        
        st.markdown("""
        L'analyse suit une hiérarchie stricte en cinq niveaux :
        
        1. **Niveau 1 - Validation** : Vérification de la validité syntaxique des SMILES générés
        2. **Niveau 2 - Nouveauté** : Comparaison avec une base de référence pour identifier les structures nouvelles
        3. **Niveau 3 - Unicité** : Élimination des doublons parmi les molécules nouvelles
        4. **Niveau 4 - Conditions** : Évaluation des molécules selon des critères spécifiques
        
        Cette approche garantit que seules les molécules passant tous les filtres sont retenues pour l'analyse finale.
        """)
        
        st.markdown('<div class="section-header">Conditions Disponibles</div>', unsafe_allow_html=True)
        
        for idx, cond in CONDITIONS.items():
            st.markdown(f"""
            <div class="metric-card" style="margin-bottom: 0.5rem;">
                <div style="font-weight: 500; color: #2C3E50;">{cond['name']}</div>
                <div style="font-size: 0.9rem; color: #5D6D7E; margin-top: 0.25rem;">{cond['description']}</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="warning-box">
            <strong>Note importante :</strong> La première exécution peut prendre quelques minutes pour charger le modèle et le vocabulaire.
            Les temps de génération et d'analyse dépendent du nombre de molécules à traiter.
        </div>
        
        <div style="text-align: center; margin-top: 2rem; color: #7F8C8D; font-size: 0.9rem;">
            Interface développée pour l'analyse hiérarchique de molécules générées - Version améliorée
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()