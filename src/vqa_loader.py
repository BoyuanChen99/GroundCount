from typing import Dict, List, Any, Union
from PIL import Image
from torch.utils.data import Dataset

ImgPath = Union[str, List[str]]

class QAImagePathDataset(Dataset):
    """
    Lazy loader for QA samples where each sample may have 1 or many images.

    Expected 'big_dict' schema:
      - image_paths: List[Union[str, List[str]]]   # per-sample: a path or a list of paths
      - question:    List[str]
      - answer:      List[str]
    """
    def __init__(self, data: Dict[str, List[Any]], image_mode: str = "RGB"):
        if "image_paths" in data:
            self.image_paths     : List[ImgPath] = data["image_paths"]
        else:
            self.image_paths     : List[ImgPath] = []
        if "images" in data:
            self.images = data["images"]
        else:
            self.images = []
        self.questions : List[str]     = data["question"]
        self.answers   : List[str]     = data["answer"]
        assert len(self.image_paths) == len(self.questions) == len(self.answers)
        self.image_mode = image_mode

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if len(self.image_paths) > 0:
            # normalize to list[str]
            image_paths = self.image_paths[idx]
            if isinstance(image_paths, str):
                image_paths = [image_paths]
            # lazy-load all images for this sample
            imgs = [Image.open(p).convert(self.image_mode) for p in image_paths]
            return {
                "images": imgs,               # List[PIL.Image.Image]
                "image_paths": image_paths,   # List[str]
                "question": self.questions[idx],
                "answer":   self.answers[idx],
            }
        else:
            return {
                "images": self.images,        # List[PIL.Image.Image]
                "image_paths": [],            # List[str]
                "question": self.questions[idx],
                "answer":   self.answers[idx],
            }

def _qa_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    # Keep nesting per sample (lists of lists) so downstream can aggregate per item.
    return {
        "images":      [b["images"] for b in batch],        # List[List[Image]]
        "image_paths": [b["image_paths"] for b in batch],   # List[List[str]]
        "question":    [b["question"] for b in batch],      # List[str]
        "answer":      [b["answer"] for b in batch],        # List[str]
    }