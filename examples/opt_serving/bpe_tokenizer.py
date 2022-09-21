class HuggingFaceByteLevelBPE:
    def __init__(self):
        try:
            from tokenizers import ByteLevelBPETokenizer
        except ImportError:
            raise ImportError(
                "Please install huggingface/tokenizers with: " "pip install tokenizers"
            )

        #bpe_vocab = file_utils.cached_path(cfg.bpe_vocab)
        #bpe_merges = file_utils.cached_path(cfg.bpe_merges)

        self.bpe = ByteLevelBPETokenizer(
            '/home/moe/opt_metaseq_125m/model/gpt2-vocab.json',
            '/home/moe/opt_metaseq_125m/model/gpt2-merges.txt',
            add_prefix_space=False
            #add_prefix_space=cfg.bpe_add_prefix_space,
        )

    def encode(self, x: str) -> str:
        return " ".join(map(str, self.bpe.encode(x).ids))

    def decode(self, x: str) -> str:
        return self.bpe.decode(
            [int(tok) if tok not in {"<unk>", "<mask>"} else tok for tok in x.split()]
        )

    def is_beginning_of_word(self, x: str) -> bool:
        return self.decode(x).startswith(" ")