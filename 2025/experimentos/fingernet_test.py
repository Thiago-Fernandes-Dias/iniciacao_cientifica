from multiprocessing.dummy import freeze_support
from fingernet.api import run_inference


def main():
    freeze_support()
    run_inference(
        input_path="./datasets/FVC/FVC2004/Dbs/DB1_A",
        output_path="./output",
        weights_path="./fingernet-updated/models/released_version/Model.pth",
        gpus=[0],
        batch_size=32,
        num_workers=1,
        recursive=True,
        mnt_degrees=True,
        compile_model=True,
        max_image_dim=1000,
        strategy="full_gpu",
        num_cpu_workers=1,
    )

if __name__ == "__main__":
    main()