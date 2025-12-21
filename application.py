#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
FLASK WEB INTERFACE FOR COMPLETE HIERARCHICAL ANALYSIS
Flask interface for hierarchical analysis of generated molecules
Academic version - Clean design with separate HTML/JS
"""

from flask import Flask, render_template, request, jsonify, send_file, session
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
import threading
import uuid
import io
import logging

logging.basicConfig(level=logging.DEBUG, filename='flask_debug.log', format='%(asctime)s - %(levelname)s - %(message)s')

# Désactiver les logs RDKit
rdBase.DisableLog('rdApp.error')

# Flask app
app = Flask(__name__)
app.secret_key = 'conditional_transformers_secret_key_2025'

# --- CONFIGURATION ---
DRIVE_PATH = '.'
VOCAB_FILE = os.path.join(DRIVE_PATH, 'vocab_dataset.json')
CHECKPOINT_FILE = os.path.join(DRIVE_PATH, '.', 'cond_gpt_categorical_extended.pth')
DATA_FILE = 's_100_str_+1M_fixed.txt'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_reference_smiles():
    """Charge le dataset de référence"""
    reference_smiles = set()
    if os.path.exists(DATA_FILE):
        print(f"Loading reference SMILES from {DATA_FILE}...")
        count = 0
        max_load = 10000  # Temporaire pour test rapide
        with open(DATA_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                if count >= max_load:
                    break
                line = line.strip()
                if line:
                    mol = Chem.MolFromSmiles(line)
                    if mol:
                        canon_smiles = Chem.MolToSmiles(mol)
                        reference_smiles.add(canon_smiles)
                count += 1
                if count % 1000 == 0:
                    print(f"Loaded {count} SMILES...")
        print(f"Total loaded: {len(reference_smiles)} unique SMILES from {count} lines")
    else:
        print(f"Warning: {DATA_FILE} not found")
    return reference_smiles

# Global variables for background tasks
task_status = {}
task_results = {}

# Load reference SMILES at startup
logging.info("Loading reference SMILES...")
reference_smiles = load_reference_smiles()
logging.info(f"Loaded {len(reference_smiles)} reference SMILES")

# --- FONCTIONS IDENTIQUES (adaptées pour Flask) ---
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

# --- FONCTIONS D'ANALYSE ADAPTÉES POUR FLASK ---
@torch.no_grad()
def load_model_and_vocab():
    """Charge le modèle et le vocabulaire"""
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

def comprehensive_hierarchical_analysis(generated_smiles, reference_smiles, task_id):
    """Analyse hiérarchique complète avec mise à jour du statut"""
    task_status[task_id] = {'status': 'running', 'message': 'Démarrage de l\'analyse hiérarchique...'}
    
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
    
    # Niveau 1: Validation
    task_status[task_id] = {'status': 'running', 'message': 'Niveau 1: Validation des SMILES...'}
    
    valid_molecules = []
    valid_smiles = []
    
    for i, smiles in enumerate(generated_smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            valid_molecules.append(mol)
            valid_smiles.append(smiles)
    
    results['total_valid'] = len(valid_molecules)
    if results['total_generated'] > 0:
        results['validity_percentage'] = (results['total_valid'] / results['total_generated']) * 100
    
    # Niveau 2: Novelty
    task_status[task_id] = {'status': 'running', 'message': 'Niveau 2: Vérification de la nouveauté...'}
    
    novel_molecules = []
    novel_smiles = []
    
    for mol, smiles in zip(valid_molecules, valid_smiles):
        canon_smiles = Chem.MolToSmiles(mol)
        if canon_smiles not in reference_smiles:
            novel_molecules.append(mol)
            novel_smiles.append(smiles)
    
    results['total_novel'] = len(novel_molecules)
    if results['total_valid'] > 0:
        results['novelty_percentage'] = (results['total_novel'] / results['total_valid']) * 100
    
    # Niveau 3: Uniqueness
    task_status[task_id] = {'status': 'running', 'message': 'Niveau 3: Vérification de l\'unicité...'}
    
    unique_novel_molecules = {}
    unique_novel_smiles = []
    
    for mol, smiles in zip(novel_molecules, novel_smiles):
        canon_smiles = Chem.MolToSmiles(mol)
        if canon_smiles not in unique_novel_molecules:
            unique_novel_molecules[canon_smiles] = mol
            unique_novel_smiles.append(smiles)
    
    results['total_unique_novel'] = len(unique_novel_molecules)
    results['unique_novel_molecules'] = unique_novel_molecules
    results['unique_novel_smiles'] = unique_novel_smiles
    
    if results['total_novel'] > 0:
        results['uniqueness_percentage'] = (results['total_unique_novel'] / results['total_novel']) * 100
    
    # Niveau 4: Conditions
    task_status[task_id] = {'status': 'running', 'message': 'Niveau 4: Évaluation des conditions...'}
    
    condition_results = {
        'total_unique_novel': results['total_unique_novel'],
        'condition_counts': {0: 0, 1: 0, 2: 0, 3: 0},
        'condition_percentages': {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0},
        'examples_per_condition': {0: [], 1: [], 2: [], 3: []},
        'properties_per_condition': {0: [], 1: [], 2: [], 3: []}
    }
    
    for condition_idx in range(4):
        condition_count = 0
        
        for canon_smiles, mol in unique_novel_molecules.items():
            if evaluate_condition(mol, condition_idx):
                condition_count += 1
                
                if len(condition_results['examples_per_condition'][condition_idx]) < 5:
                    condition_results['examples_per_condition'][condition_idx].append(canon_smiles)
                    
                    props = {
                        'SMILES': canon_smiles,
                        'LogP': Crippen.MolLogP(mol),
                        'MW': Descriptors.MolWt(mol),
                        'HBD': Lipinski.NumHDonors(mol),
                        'HBA': Lipinski.NumHAcceptors(mol),
                        'RotB': Lipinski.NumRotatableBonds(mol),
                    }
                    # Handle NaN values
                    import math
                    for key in props:
                        if key != 'SMILES' and isinstance(props[key], float) and math.isnan(props[key]):
                            props[key] = None
                    
                    if condition_idx in [1, 3]:
                        aromatic = Lipinski.NumAromaticRings(mol)
                        aliphatic = Lipinski.NumAliphaticRings(mol)
                        r_val = calculate_r_value(mol)
                        if isinstance(aromatic, float) and math.isnan(aromatic): aromatic = None
                        if isinstance(aliphatic, float) and math.isnan(aliphatic): aliphatic = None
                        if isinstance(r_val, float) and math.isnan(r_val): r_val = None
                        props.update({
                            'Aromatic': aromatic,
                            'Aliphatic': aliphatic,
                            'R-value': r_val,
                            'Functional': "Oui" if has_functional_group(mol) else "Non"
                        })
                    condition_results['properties_per_condition'][condition_idx].append(props)
        
        condition_results['condition_counts'][condition_idx] = condition_count
        if results['total_unique_novel'] > 0:
            condition_results['condition_percentages'][condition_idx] = (condition_count / results['total_unique_novel']) * 100
    
    results['condition_results'] = condition_results
    
    # Remove Mol objects to allow JSON serialization
    del results['unique_novel_molecules']
    
    task_status[task_id] = {'status': 'completed', 'message': 'Analyse hiérarchique terminée !'}
    task_results[task_id] = results
    return results

@torch.no_grad()
def generate_molecules(model, condition_tensor, stoi, itos, start_idx, end_idx, 
                      num_molecules=10000, temperature=0.6, task_id=None):
    """Génère des molécules avec mise à jour du statut"""
    if task_id:
        task_status[task_id] = {'status': 'running', 'message': f'Génération de {num_molecules} molécules...'}
    
    generated_smiles = []
    
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
    
    if task_id:
        task_status[task_id] = {'status': 'completed', 'message': f'{len(generated_smiles)} molécules générées avec succès !'}
    
    return generated_smiles

def calculate_intdiv(smiles_list):
    """Calcule la diversité interne"""
    if len(smiles_list) < 2:
        return 0.0
    
    sample_size = min(2000, len(smiles_list))
    if len(smiles_list) > sample_size:
        indices = np.random.choice(len(smiles_list), sample_size, replace=False)
        sample_smiles = [smiles_list[i] for i in indices]
    else:
        sample_smiles = smiles_list
    
    fingerprints = []
    for smiles in sample_smiles:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
            fingerprints.append(fp)
    
    if len(fingerprints) < 2:
        return 0.0
    
    similarities = []
    for i in range(len(fingerprints)):
        sims = DataStructs.BulkTanimotoSimilarity(fingerprints[i], fingerprints[i+1:])
        similarities.extend(sims)
    
    if similarities:
        mean_similarity = np.mean(similarities)
        intdiv = 1 - mean_similarity
        return intdiv
    else:
        return 0.0

def create_visualization(results):
    """Crée des visualisations et retourne les données JSON"""
    categories = ['Générées', 'Valides', 'Nouvelles', 'Uniques']
    counts = [
        results['total_generated'],
        results['total_valid'],
        results['total_novel'],
        results['total_unique_novel']
    ]
    
    percentages = [
        100.0,
        results['validity_percentage'],
        results['novelty_percentage'],
        results['uniqueness_percentage']
    ]
    
    condition_names = [CONDITIONS[i]["name"] for i in range(4)]
    condition_counts = [results['condition_results']['condition_counts'][i] for i in range(4)]
    condition_percentages = [results['condition_results']['condition_percentages'][i] for i in range(4)]
    
    return {
        'hierarchy': {
            'categories': categories,
            'counts': counts,
            'percentages': percentages
        },
        'conditions': {
            'names': condition_names,
            'counts': condition_counts,
            'percentages': condition_percentages
        }
    }

# --- ROUTES FLASK ---
@app.route('/')
def index():
    return render_template('index.html', conditions=CONDITIONS)

@app.route('/generate', methods=['POST'])
def generate():
    try:
        data = request.get_json()
        condition_option = int(data.get('condition', 2))
        num_molecules = int(data.get('num_molecules', 10000))
        temperature = float(data.get('temperature', 0.6))
        calculate_intdiv_option = data.get('calculate_intdiv', True)
        
        # Créer un ID de tâche unique
        task_id = str(uuid.uuid4())
        task_status[task_id] = {'status': 'starting', 'message': 'Initialisation...'}
        
        # Lancer la génération en arrière-plan
        def background_task():
            logging.info(f"Task {task_id} started")
            try:
                # Chargement des ressources
                task_status[task_id] = {'status': 'running', 'message': 'Chargement du modèle...'}
                logging.info("Loading model")
                model, stoi, itos = load_model_and_vocab()
                logging.info("Model loaded successfully")
                start_token = stoi['<start>']
                end_token = stoi['<end>']
                
                # Condition
                condition_vector = CONDITIONS[condition_option]["get_vector"](None)
                condition_tensor = torch.tensor([condition_vector], dtype=torch.float32)
                
                # Génération
                logging.info(f"Starting generation of {num_molecules} molecules")
                generated_smiles = generate_molecules(
                    model, condition_tensor, stoi, itos, start_token, end_token,
                    num_molecules=num_molecules, temperature=temperature, task_id=task_id
                )
                logging.info(f"Generated {len(generated_smiles)} smiles")
                
                # Analyse
                logging.info("Loading reference smiles")
                logging.info(f"Loaded {len(reference_smiles)} reference smiles")
                logging.info("Starting hierarchical analysis")
                results = comprehensive_hierarchical_analysis(generated_smiles, reference_smiles, task_id)
                logging.info("Analysis completed")
                
                # Diversité
                if calculate_intdiv_option and results['total_unique_novel'] > 0:
                    logging.info("Calculating intdiv")
                    intdiv = calculate_intdiv(results['unique_novel_smiles'])
                    results['intdiv'] = intdiv
                    logging.info(f"Intdiv calculated: {intdiv}")
                
                # Visualisations
                logging.info("Creating visualizations")
                results['visualizations'] = create_visualization(results)
                logging.info("Visualizations created")
                
                logging.info("Setting task_results")
                task_status[task_id] = {'status': 'completed', 'message': 'Analyse hiérarchique terminée !'}
                task_results[task_id] = results
                logging.info("Task completed successfully")
                
            except Exception as e:
                logging.error(f"Exception in background task: {e}", exc_info=True)
                task_status[task_id] = {'status': 'error', 'message': str(e)}
        
        thread = threading.Thread(target=background_task)
        thread.daemon = True
        thread.start()
        
        return jsonify({'task_id': task_id, 'status': 'started'})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/status/<task_id>')
def get_status(task_id):
    status = task_status.get(task_id, {'status': 'not_found', 'message': 'Tâche non trouvée'})
    return jsonify(status)

@app.route('/results/<task_id>')
def get_results(task_id):
    if task_id in task_results:
        return jsonify(task_results[task_id])
    else:
        return jsonify({'error': 'Résultats non disponibles'}), 404

@app.route('/download/<task_id>/<file_type>')
def download_file(task_id, file_type):
    if task_id not in task_results:
        return jsonify({'error': 'Résultats non disponibles'}), 404
    
    results = task_results[task_id]
    
    if file_type == 'json':
        # Export JSON
        export_data = {
            'parametres': {
                'condition': 'Condition loaded',
                'temperature': 'Temperature loaded',
                'nombre_molecules': results['total_generated']
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
            export_data['resultats_conditions'][str(cond_idx)] = {
                'nom': CONDITIONS[cond_idx]['name'],
                'satisfaites': results['condition_results']['condition_counts'][cond_idx],
                'pourcentage': results['condition_results']['condition_percentages'][cond_idx]
            }
        
        json_str = json.dumps(export_data, indent=2, ensure_ascii=False)
        return send_file(
            io.BytesIO(json_str.encode('utf-8')),
            mimetype='application/json',
            as_attachment=True,
            download_name=f'analyse_molecules_{time.strftime("%Y%m%d_%H%M%S")}.json'
        )
    
    elif file_type == 'smiles':
        if results['unique_novel_smiles']:
            smiles_text = '\n'.join(results['unique_novel_smiles'])
            return send_file(
                io.BytesIO(smiles_text.encode('utf-8')),
                mimetype='text/plain',
                as_attachment=True,
                download_name=f'smiles_uniques_{time.strftime("%Y%m%d_%H%M%S")}.smi'
            )
    
    return jsonify({'error': 'Type de fichier non supporté'}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5005)