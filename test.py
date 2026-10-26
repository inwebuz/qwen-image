from pathlib import Path
from uuid import uuid4

import torch
from diffusers import QwenImage21Pipeline


def main():
    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is unavailable. Install a CUDA-enabled PyTorch build compatible "
            "with your GPU; see README.md."
        )

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported(including_emulation=False) else torch.float16
    print(f"GPU: {torch.cuda.get_device_name(0)}; dtype: {dtype}")
    print("Sequential CPU offload enabled. Loading requires substantial system RAM.")

    try:
        pipe = QwenImage21Pipeline.from_pretrained(
            "Qwen/Qwen-Image-2.1",
            dtype=dtype,
            low_cpu_mem_usage=True,
            disable_mmap=False,
        )
    except (MemoryError, OSError) as error:
        cause = error
        while cause.__context__ is not None:
            cause = cause.__context__
        if isinstance(error, MemoryError) or getattr(cause, "winerror", None) == 1455:
            raise SystemExit(
                "Not enough system memory to load model weights. CPU offload starts only "
                "after loading. Close memory-heavy applications and increase the Windows "
                "paging file; see README.md. "
                f"Original error: {type(cause).__name__}: {cause}"
            ) from error
        raise
    pipe.vae.enable_tiling()
    pipe.enable_sequential_cpu_offload()

    try:
        image = pipe(
            prompt='Print "Happy New Year" with transparent background',
            width=512,
            height=512,
            num_inference_steps=40,
            generator=torch.Generator("cpu").manual_seed(42),
        ).images[0]
    except torch.cuda.OutOfMemoryError as error:
        raise SystemExit(
            "GPU memory exhausted. Close other GPU applications or reduce width and height."
        ) from error

    output_dir = Path(__file__).resolve().parent / "storage"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"t2i_{uuid4().hex}.png"
    image.save(output_path)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
