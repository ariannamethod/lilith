from typing import List
import json
import os


class Tokenizer:
    """
    Llama tokenizer wrapper.
    
    Tour 3: Enhanced with error handling and validation.
    """
    
    def __init__(self, model_path: str):
        """
        Initialize tokenizer from model file.
        
        Args:
            model_path: Path to tokenizer JSON file
        
        Raises:
            FileNotFoundError: If tokenizer file doesn't exist
            ValueError: If tokenizer format is invalid
        """
        # Tour 3: Check file existence
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Tokenizer file not found: {model_path}\n"
                f"Lilith cannot speak without her vocabulary."
            )
        
        try:
            with open(model_path, "r", encoding="utf-8") as f:
                model = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Invalid JSON in tokenizer file {model_path}: {e}\n"
                f"Lilith's vocabulary is corrupted."
            )
        except UnicodeDecodeError as e:
            raise ValueError(
                f"Encoding error in tokenizer file {model_path}: {e}\n"
                f"File must be UTF-8 encoded."
            )
        
        # Tour 3: Validate required fields
        if 'tokens' not in model:
            raise ValueError("Tokenizer file missing 'tokens' field")
        if 'scores' not in model:
            raise ValueError("Tokenizer file missing 'scores' field")
        
        self.vocab = model["tokens"]
        self.scores = model["scores"]
        
        # Tour 3: Validate vocab and scores match
        if len(self.vocab) != len(self.scores):
            raise ValueError(
                f"Tokenizer vocab size ({len(self.vocab)}) doesn't match "
                f"scores size ({len(self.scores)})"
            )
        
        self.bos_id = 1
        self.eos_id = 2

    def str_lookup(self, token: str) -> int:
        try:
            index = self.vocab.index(token)
            return index
        except ValueError as err:
            return -1

    def encode(
            self,
            text: str,
            add_bos: bool = True,
            add_eos: bool = False,
    ) -> List[int]:
        """
        Encode text to token IDs.
        
        Args:
            text: Input text
            add_bos: Whether to add beginning-of-sequence token
            add_eos: Whether to add end-of-sequence token
        
        Returns:
            List of token IDs
        """
        # Tour 3: Handle edge cases
        if not text:
            tokens = []
            if add_bos:
                tokens.append(self.bos_id)
            if add_eos:
                tokens.append(self.eos_id)
            return tokens
        
        # Tour 3: Handle Unicode gracefully
        try:
            tokens = []
            for pos, char in enumerate(text):
                id = self.str_lookup(char)
                if id >= 0:
                    tokens.append(id)
                # If character not in vocab, skip it (graceful degradation)
        except Exception as e:
            # If encoding fails catastrophically, return safe fallback
            print(f"Warning: Encoding error for text: {e}")
            tokens = []
        while True:
            best_score = -1e10
            best_id = -1
            best_idx = -1

            for i in range(len(tokens) - 1):
                # Check if we can merge the pair (tokens[i], tokens[i+1])
                string = self.vocab[tokens[i]] + self.vocab[tokens[i + 1]]
                id = self.str_lookup(string)
                if id != -1 and self.scores[id] > best_score:
                    best_score = self.scores[id]
                    best_id = id
                    best_idx = i

            if best_idx == -1:
                break

            # Merge the consecutive pair (best_idx, best_idx+1) into new token best_id
            tokens[best_idx] = best_id
            # Delete token at position best_idx+1, shift the entire sequence back 1
            tokens = tokens[0: best_idx + 1] + tokens[best_idx + 2:]
        if add_bos:
            tokens.insert(0, self.bos_id)
        if add_eos:
            tokens.append(self.eos_id)
        return tokens

    def decode(self, ids: List[int]) -> str:
        """
        Decode token IDs to text.
        
        Args:
            ids: List of token IDs
        
        Returns:
            Decoded text string
        """
        # Tour 3: Handle empty input
        if not ids:
            return ""
        
        res = []
        for i in ids:
            # Tour 3: Bounds checking
            if 0 <= i < len(self.vocab):
                token = self.vocab[i]
                res.append(token)
            # Skip invalid token IDs gracefully
        
        text = "".join(res)
        text = text.strip("<s>").strip("</s>")
        return text
