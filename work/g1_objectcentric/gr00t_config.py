"""Custom fixed-base palms; deliberately NEW_EMBODIMENT, not REAL_G1.

Use in an installed GR00T environment. Absolute poses are stored on disk;
GR00T's processor performs its relative EEF conversion. Our local body-frame
6D ActionChunk convention must not be assumed to equal that conversion.
"""
from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import ModalityConfig, ActionConfig, ActionRepresentation, ActionType, ActionFormat

config = {
    "video": ModalityConfig(delta_indices=[0], modality_keys=["front", "side"]),
    "state": ModalityConfig(delta_indices=[0], modality_keys=["left_eef", "right_eef", "arms"]),
    "action": ModalityConfig(delta_indices=list(range(40)), modality_keys=["left_eef", "right_eef", "arms"],
        action_configs=[
            ActionConfig(rep=ActionRepresentation.RELATIVE, type=ActionType.EEF, format=ActionFormat.XYZ_ROTVEC),
            ActionConfig(rep=ActionRepresentation.RELATIVE, type=ActionType.EEF, format=ActionFormat.XYZ_ROTVEC),
            ActionConfig(rep=ActionRepresentation.RELATIVE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),
        ]),
    "language": ModalityConfig(delta_indices=[0], modality_keys=["annotation.human.task_description"]),
}
register_modality_config(config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
