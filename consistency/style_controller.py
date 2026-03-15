class StyleController:
    def __init__(self, style_preset: str) -> None:
        self.style_preset = style_preset

    def apply(self, prompt: str) -> str:
        return f"{prompt}, style: {self.style_preset}"
