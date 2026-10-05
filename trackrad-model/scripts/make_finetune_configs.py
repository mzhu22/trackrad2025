"""Generate the 15 SAM2 fine-tuning configs used in the paper.

5 starting models x 3 training sets (manual, semiauto, combined), all rendered
from scripts/finetune_template.yaml into
sam2/sam2/configs/sam2.1_training/<model>_<dataset>_finetune.yaml.

Run with: uv run python scripts/make_finetune_configs.py
(from trackrad-model/). Expects the data layout produced by
scripts/prepare_sam2_finetune_data.py and checkpoints in resources/.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = ROOT / "scripts" / "finetune_template.yaml"
OUT_DIR = ROOT / "sam2" / "sam2" / "configs" / "sam2.1_training"
DATA_ROOT = ROOT.parent / "data" / "sam2_finetune"
RESOURCES = ROOT / "resources"

TINY_TRUNK = """\
                embed_dim: 96
                num_heads: 1
                stages: [1, 2, 7, 2]
                global_att_blocks: [5, 7, 9]
                window_pos_embed_bkg_spatial_size: [7, 7]
"""
SMALL_TRUNK = """\
                embed_dim: 96
                num_heads: 1
                stages: [1, 2, 11, 2]
                global_att_blocks: [7, 10, 13]
                window_pos_embed_bkg_spatial_size: [7, 7]
"""
BASE_PLUS_TRUNK = """\
                embed_dim: 112
                num_heads: 2
"""
LARGE_TRUNK = """\
                embed_dim: 144
                num_heads: 2
                stages: [2, 6, 36, 4]
                global_att_blocks: [23, 33, 43]
                window_pos_embed_bkg_spatial_size: [7, 7]
                window_spec: [8, 4, 16, 8]
"""

# name -> (trunk, FPN channels, resolution, checkpoint). MedSAM2 is a Tiny
# architecture that runs at 512x512 instead of 1024x1024.
MODELS = {
    "sam2.1_hiera_t": (TINY_TRUNK, [768, 384, 192, 96], 1024, "sam2.1_hiera_tiny.pt"),
    "sam2.1_hiera_s": (SMALL_TRUNK, [768, 384, 192, 96], 1024, "sam2.1_hiera_small.pt"),
    "sam2.1_hiera_b+": (
        BASE_PLUS_TRUNK,
        [896, 448, 224, 112],
        1024,
        "sam2.1_hiera_base_plus.pt",
    ),
    "sam2.1_hiera_l": (
        LARGE_TRUNK,
        [1152, 576, 288, 144],
        1024,
        "sam2.1_hiera_large.pt",
    ),
    "sam2.1_medsam2": (TINY_TRUNK, [768, 384, 192, 96], 512, "MedSAM2_latest.pt"),
}
DATASETS = ["manual", "semiauto", "combined"]


def main() -> None:
    template = TEMPLATE.read_text()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for model, (trunk, channels, resolution, ckpt) in MODELS.items():
        feat = resolution // 16
        for dataset in DATASETS:
            text = (
                template.replace("@RESOLUTION@", str(resolution))
                .replace("@TRUNK@", trunk)
                .replace("@CHANNELS@", str(channels))
                .replace("@FEAT@", f"[{feat}, {feat}]")
                .replace("@DATA_ROOT@", str(DATA_ROOT))
                .replace("@FILE_LIST@", dataset)
                .replace("@CKPT@", str(RESOURCES / ckpt))
            )
            assert "@" not in text.replace("@package", "")
            out = OUT_DIR / f"{model}_{dataset}_finetune.yaml"
            out.write_text(text)
            print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
