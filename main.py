import torch
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import Dict, Callable
from enum import Enum
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

class EditingTechnique(Enum):
    DARE = "DARE"
    BITDELTA = "BitDelta"
    EXPO = "EXPO"

class DeltaEditor:
    """Class to handle delta parameter editing techniques."""

    def __init__(self, technique: EditingTechnique, **kwargs) -> None:
        self.technique = technique
        self.params = kwargs
        # Map techniques to their corresponding editing functions
        self.edit_functions: Dict[EditingTechnique, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = {
            EditingTechnique.DARE: self.dare_editing,
            EditingTechnique.BITDELTA: self.bitdelta_editing,
            EditingTechnique.EXPO: self.expo_editing,
        }

        if self.technique not in self.edit_functions:
            raise ValueError(f"Unsupported editing technique: {self.technique}")
    @staticmethod
    def calculate_delta(W_pre: torch.Tensor, W_post: torch.Tensor) -> torch.Tensor:
        """Calculate the difference between post and pre parameters."""
        return W_post - W_pre
    def dare_editing(self, W_pre: torch.Tensor, W_post: torch.Tensor) -> torch.Tensor:
        """DARE technique with random drop and rescale on delta parameters."""
        p: float = self.params.get('p', 0.1)
        delta_W = self.calculate_delta(W_pre, W_post)
        M = torch.bernoulli((1 - p) * torch.ones_like(delta_W))  # Bernoulli mask
        scaled_delta_W = (1 / (1 - p)) * M * delta_W
        W_dare = W_pre + scaled_delta_W
        logger.debug("Applied DARE editing.")
        return W_dare

    def bitdelta_editing(self, W_pre: torch.Tensor, W_post: torch.Tensor) -> torch.Tensor:
        """BitDelta technique with sign and mean magnitude scaling on delta parameters."""
        delta_W = self.calculate_delta(W_pre, W_post)
        avg_magnitude = torch.mean(torch.abs(delta_W))
        if avg_magnitude == 0:
            quantized_delta_W = torch.zeros_like(delta_W)
            logger.warning("Average magnitude is zero in BitDelta editing.")
        else:
            quantized_delta_W = avg_magnitude * torch.sign(delta_W)
        W_bitdelta = W_pre + quantized_delta_W
        logger.debug("Applied BitDelta editing.")
        return W_bitdelta
    def expo_editing(self, W_pre: torch.Tensor, W_post: torch.Tensor) -> torch.Tensor:
        """EXPO technique with extrapolation on delta parameters."""
        alpha: float = self.params.get('alpha', 0.5)
        delta_W = self.calculate_delta(W_pre, W_post)
        scaled_delta_W = (1 + alpha) * delta_W
        W_expo = W_pre + scaled_delta_W
        logger.debug("Applied EXPO editing.")
        return W_expo
    def edit_parameters(self, W_pre: torch.Tensor, W_post: torch.Tensor) -> torch.Tensor:
        """Apply the selected editing technique to the parameters."""
        edit_func = self.edit_functions[self.technique]
        return edit_func(W_pre, W_post)
def load_model_and_tokenizer(model_name: str) -> (AutoModelForSequenceClassification, AutoTokenizer): # type: ignore
    """Load the pre-trained model and tokenizer."""
    try:
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        logger.info(f"Loaded model and tokenizer for '{model_name}'.")
        return model, tokenizer
    except Exception as e:
        logger.error(f"Error loading model or tokenizer: {e}")
        raise
def simulate_post_training(model: AutoModelForSequenceClassification, seed: int = 42) -> Dict[str, torch.Tensor]:
    """Simulate post-training by slightly altering the model's parameters."""
    torch.manual_seed(seed)
    posttrained_params: Dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for name, param in model.named_parameters():
            posttrained_params[name] = param.clone().detach()
            param.data += torch.randn_like(param) * 0.01 
    logger.info("Simulated post-training parameter updates.")
    return posttrained_params

def test_delta_editing(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    editor: DeltaEditor,
    text: str
) -> None:
    """Test delta parameter editing on a sample input."""
    inputs = tokenizer(text, return_tensors="pt")
    logger.info(f"Testing on input text: '{text}'")

    # Original model predictions
    with torch.no_grad():
        original_outputs = model(**inputs).logits
    logger.info(f"Original Output: {original_outputs}")

    # Clone pre-trained parameters
    pretrained_params: Dict[str, torch.Tensor] = {name: param.clone() for name, param in model.named_parameters()}
    logger.debug("Cloned pre-trained parameters.")

    # Simulate post-training
    posttrained_params = simulate_post_training(model)

    # Apply the selected editing technique
    for name, param in model.named_parameters():
        W_pre = pretrained_params[name]
        W_post = posttrained_params[name]

        try:
            W_edited = editor.edit_parameters(W_pre, W_post)
            param.data.copy_(W_edited)
            logger.debug(f"Updated parameter '{name}' using {editor.technique.value}.")
        except Exception as e:
            logger.error(f"Error editing parameter '{name}': {e}")
            raise

    # Edited model predictions
    with torch.no_grad():
        edited_outputs = model(**inputs).logits
    logger.info(f"Edited Output: {edited_outputs}")

def main() -> None:
    """Main function to execute the delta parameter editing test."""
    model_name = "distilbert-base-uncased"
    model, tokenizer = load_model_and_tokenizer(model_name)
    technique = EditingTechnique.EXPO  # Change to DARE or BITDELTA as needed
    editor_params = {'alpha': 0.5} if technique == EditingTechnique.EXPO else {'p': 0.1}
    editor = DeltaEditor(technique=technique, **editor_params)
    logger.info(f"Selected editing technique: {technique.value}")

    # Sample input text
    text = "Delta parameter editing can be complex, but it is useful."

    # Run the test
    test_delta_editing(model, tokenizer, editor, text)

if __name__ == "__main__":
    main()










