"""
Embedding extraction module for LLMEmbed reproduction.
Loads lightweight LLMs and extracts embeddings from hidden states.
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from typing import List, Optional, Union
import numpy as np
from tqdm import tqdm
import config


class LLMEmbedder:
    """
    Extracts embeddings from lightweight LLMs following LLMEmbed methodology.
    Supports multiple extraction methods: mean pooling, CLS token, last token.
    """
    
    def __init__(
        self,
        model_name: str = None,
        use_quantization: bool = None,
        quantization_bits: int = None,
        device: str = None
    ):
        """
        Initialize the embedder.
        
        Args:
            model_name: Name of the model to load
            use_quantization: Whether to use quantization
            quantization_bits: Number of bits for quantization (4 or 8)
            device: Device to run on ('cuda' or 'cpu')
        """
        self.model_name = model_name or config.MODEL_NAME
        self.use_quantization = use_quantization if use_quantization is not None else config.USE_QUANTIZATION
        self.quantization_bits = quantization_bits or config.QUANTIZATION_BITS
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Get model path
        self.model_path = config.MODEL_PATH_MAP.get(self.model_name, self.model_name)
        
        # Load tokenizer and model
        self.tokenizer = None
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the tokenizer and model with optional quantization."""
        print(f"Loading model: {self.model_path}")
        print(f"Device: {self.device}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )
        
        # Set padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Configure quantization if needed
        quantization_config = None
        if self.use_quantization and self.device == 'cuda':
            if self.quantization_bits == 4:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
            elif self.quantization_bits == 8:
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True
                )
        
        # Load model
        if quantization_config:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32
            )
            self.model = self.model.to(self.device)
        
        self.model.eval()
        print(f"Model loaded successfully. Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def extract_embeddings(
        self,
        texts: List[str],
        extraction_method: str = None,
        layer_indices: Optional[List[int]] = None,
        batch_size: int = None,
        max_length: int = None
    ) -> np.ndarray:
        """
        Extract embeddings from texts using the specified method.
        
        Args:
            texts: List of input texts
            extraction_method: Method to extract embeddings ('mean_pooling', 'cls_token', 'last_token')
            layer_indices: Specific layers to extract from (None = all layers)
            batch_size: Batch size for processing
            max_length: Maximum sequence length
            
        Returns:
            numpy array of shape (n_texts, embedding_dim)
        """
        extraction_method = extraction_method or config.EXTRACTION_METHOD
        batch_size = batch_size or config.BATCH_SIZE
        max_length = max_length or config.MAX_SEQUENCE_LENGTH
        
        all_embeddings = []
        
        # Process in batches
        for i in tqdm(range(0, len(texts), batch_size), desc="Extracting embeddings"):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )
            
            # Move to device
            input_ids = encoded['input_ids'].to(self.device)
            attention_mask = encoded['attention_mask'].to(self.device)
            
            # Extract hidden states
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True
                )
            
            # Get hidden states (tuple of tensors, one per layer)
            hidden_states = outputs.hidden_states  # List of [batch_size, seq_len, hidden_dim]
            
            # Select layers if specified
            if layer_indices is not None:
                hidden_states = [hidden_states[i] for i in layer_indices]
            
            # Apply extraction method
            batch_embeddings = self._extract_from_hidden_states(
                hidden_states,
                attention_mask,
                extraction_method
            )
            
            all_embeddings.append(batch_embeddings.cpu().numpy())
        
        # Concatenate all batches
        embeddings = np.vstack(all_embeddings)
        return embeddings
    
    def _extract_from_hidden_states(
        self,
        hidden_states: List[torch.Tensor],
        attention_mask: torch.Tensor,
        extraction_method: str
    ) -> torch.Tensor:
        """
        Extract sentence embeddings from hidden states.
        
        Args:
            hidden_states: List of hidden state tensors from different layers
            attention_mask: Attention mask tensor
            extraction_method: Method to use
            
        Returns:
            Tensor of shape (batch_size, embedding_dim)
        """
        # Use the last layer's hidden states for extraction
        last_hidden = hidden_states[-1]  # [batch_size, seq_len, hidden_dim]
        
        if extraction_method == "mean_pooling":
            # Mean pooling over sequence length, weighted by attention mask
            mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
            sum_hidden = torch.sum(last_hidden * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            embeddings = sum_hidden / sum_mask
            
        elif extraction_method == "cls_token":
            # Use first token (CLS-equivalent)
            embeddings = last_hidden[:, 0, :]
            
        elif extraction_method == "last_token":
            # Use last non-padding token
            seq_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden.size(0)
            embeddings = last_hidden[torch.arange(batch_size), seq_lengths, :]
            
        else:
            raise ValueError(f"Unknown extraction method: {extraction_method}")
        
        return embeddings
    
    def extract_multi_layer_embeddings(
        self,
        texts: List[str],
        layer_indices: Optional[List[int]] = None,
        batch_size: int = None,
        max_length: int = None
    ) -> List[np.ndarray]:
        """
        Extract embeddings from multiple layers for fusion.
        
        Args:
            texts: List of input texts
            layer_indices: Specific layers to extract (None = all layers)
            batch_size: Batch size for processing
            max_length: Maximum sequence length
            
        Returns:
            List of numpy arrays, one per layer
        """
        batch_size = batch_size or config.BATCH_SIZE
        max_length = max_length or config.MAX_SEQUENCE_LENGTH
        
        all_layer_embeddings = []
        
        # Process in batches
        for i in tqdm(range(0, len(texts), batch_size), desc="Extracting multi-layer embeddings"):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )
            
            input_ids = encoded['input_ids'].to(self.device)
            attention_mask = encoded['attention_mask'].to(self.device)
            
            # Extract hidden states
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True
                )
            
            hidden_states = outputs.hidden_states
            
            # Select layers if specified
            if layer_indices is not None:
                hidden_states = [hidden_states[i] for i in layer_indices]
            
            # Extract from each layer using mean pooling
            batch_layer_embeddings = []
            for layer_hidden in hidden_states:
                mask_expanded = attention_mask.unsqueeze(-1).expand(layer_hidden.size()).float()
                sum_hidden = torch.sum(layer_hidden * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                layer_emb = sum_hidden / sum_mask
                batch_layer_embeddings.append(layer_emb.cpu().numpy())
            
            if not all_layer_embeddings:
                all_layer_embeddings = [[] for _ in range(len(batch_layer_embeddings))]
            
            for idx, layer_emb in enumerate(batch_layer_embeddings):
                all_layer_embeddings[idx].append(layer_emb)
        
        # Concatenate batches for each layer
        final_embeddings = [np.vstack(layer_embs) for layer_embs in all_layer_embeddings]
        return final_embeddings

