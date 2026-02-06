
import tiktoken



class Tokenizer:
    def __init__(self, model="cl100k_base", max_tokens=8191):
        self._enc = tiktoken.get_encoding(model)
        self.max_tokens = max_tokens

    def tokenize(self, text):
        return self._enc.encode(text)
    
    def count_tokens(self, text):
        return len(self.tokenize(text))
    
    def trim_text(self, text):
        tokens = self.tokenize(text)
        return self._enc.decode(tokens[:self.max_tokens])
    
    def trim_texts(self, texts):
        return [self.trim_text(text) for text in texts]
        
    def detokenize(self, tokens):
        return self._enc.decode(tokens)