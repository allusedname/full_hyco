from hycoclip.config import LazyCall as L
from hycoclip.evaluation.hierarchical_metrics import HierarchicalMetricsEvaluator


evaluator = L(HierarchicalMetricsEvaluator)(
    datasets_and_prompts={
        "imagenet": [
            "i took a picture : itap of a {}.",
            "pics : a bad photo of the {}.",
            "pics : a origami {}.",
            "pics : a photo of the large {}.",
            "pics : a {} in a video game.",
            "pics : art of the {}.",
            "pics : a photo of the small {}.",
        ]
    },
    data_dir="/scratch-shared/apal/meru_eval",
    image_size=224
)
