#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
TEST SCRIPT FOR MOLECULE GENERATION
Inspired by app.py - Simple test to generate molecules
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from dataclasses import dataclass
import os
import json
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, DataStructs, Crippen, Lipinski
from rdkit import rdBase

# Désactiver les logs RDKit
rdBase.DisableLog('rdApp.error')

# --- CONFIGURATION ---
DRIVE_PATH = '.'
VOCAB_FILE = os.path.join(DRIVE_PATH, 'vocab_dataset.json')
CHECKPOINT_FILE = os.path.join(DRIVE_PATH, '.', 'cond_gpt_categorical_extended.pth')
DATA_FILE = 's_100_str_+1M_fixed.txt'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- FONCTIONS IDENTIQUES ---
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

# --- FONCTIONS D'ANALYSE ---
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

@torch.no_grad()
def generate_molecules(model, condition_tensor, stoi, itos, start_idx, end_idx, 
                      num_molecules=10, temperature=0.6):
    """Génère des molécules"""
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
    
    return generated_smiles

# --- TEST SCRIPT ---
def main():
    print("Test de génération de molécules")
    print("=" * 50)
    
    try:
        print("Chargement du modèle...")
        model, stoi, itos = load_model_and_vocab()
        print("Modèle chargé avec succès !")
        
        start_token = stoi['<start>']
        end_token = stoi['<end>']
        print(f"Start token: {start_token}, End token: {end_token}")
        
        # Condition 2 (Structural)
        condition_option = 2
        condition_vector = CONDITIONS[condition_option]["get_vector"](None)
        condition_tensor = torch.tensor([condition_vector], dtype=torch.float32)
        print(f"Condition: {CONDITIONS[condition_option]['name']}")
        print(f"Vector: {condition_vector}")
        
        num_molecules = 10
        temperature = 0.6
        print(f"Génération de {num_molecules} molécules avec température {temperature}...")
        
        generated_smiles = generate_molecules(
            model, condition_tensor, stoi, itos, start_token, end_token,
            num_molecules=num_molecules, temperature=temperature
        )
        
        print(f"\n{len(generated_smiles)} molécules générées:")
        print("-" * 30)
        
        valid_count = 0
        for i, smiles in enumerate(generated_smiles, 1):
            mol = Chem.MolFromSmiles(smiles)
            is_valid = mol is not None
            if is_valid:
                valid_count += 1
            status = "✓" if is_valid else "✗"
            print(f"{i:2d}. {status} {smiles}")
        
        print("-" * 30)
        print(f"Valides: {valid_count}/{len(generated_smiles)} ({valid_count/len(generated_smiles)*100:.1f}%)")
        
    except Exception as e:
        print(f"Erreur: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()