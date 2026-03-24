import logging
from multiprocessing.dummy import freeze_support

from torch.nn.functional import threshold

from flx.benchmarks.matchers import CosineSimilarityMatcher
from flx.data.dataset import Dataset
from flx.data.embedding_loader import EmbeddingLoader
from flx.data.image_helpers import pad_and_resize_to_deepprint_input_size
from flx.data.image_loader import FVC2004Loader
from flx.data.transformed_image_loader import TransformedImageLoader
from flx.extractor.fixed_length_extractor import (
    DeepPrintExtractor,
    get_DeepPrint_TexMinu,
)
from flx.image_processing.binarization import LazilyAllocatedBinarizer
from flx.scripts.generate_benchmarks import create_verification_benchmark
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO)


def run_benchmark(db_path: str, subjects: list[int], impressions_per_subject: list[int]):
    deep_print_tex_extractor: DeepPrintExtractor = get_DeepPrint_TexMinu(
        num_training_subjects=8000, num_dims=256
    )
    deep_print_tex_extractor.load_best_model(
        "./fixed-length-fingerprint-extractors/models"
    )

    image_loader = TransformedImageLoader(
        images=FVC2004Loader(db_path),
        poses=None,
        transforms=[
            LazilyAllocatedBinarizer(5.0),
            pad_and_resize_to_deepprint_input_size,
        ],
    )

    test_dataset: Dataset = Dataset(image_loader, image_loader.ids)
    tex_embeddings, minutia_embeddings = deep_print_tex_extractor.extract(test_dataset)

    benchmark = create_verification_benchmark(
        subjects=subjects,
        impressions_per_subject=impressions_per_subject
    )

    matcher = CosineSimilarityMatcher(
        EmbeddingLoader.combine(tex_embeddings, minutia_embeddings)
    )

    results = benchmark.run(matcher)

    return results


def generate_graph(xs, ys, metric, name):
    plt.figure()
    plt.plot(xs, ys, label=metric)
    plt.xlabel("Limiar de corte")
    plt.ylabel("Taxa")
    plt.title(f"{metric} x Limiar de corte")
    plt.legend()
    plt.grid()
    plt.savefig(f"{name}.png")


def main():
    freeze_support()
    base_path = "datasets/FVC/FVC2004/Dbs/"
    a_subjects = list(range(100))
    b_subjects = list(range(101, 110))
    impressions_per_subject = list(range(8))
    db1_a_result = run_benchmark(base_path + "DB1_A", a_subjects, impressions_per_subject)
    db1_b_result = run_benchmark(base_path + "DB1_B", b_subjects, impressions_per_subject)
    db2_a_result = run_benchmark(base_path + "DB2_A", a_subjects, impressions_per_subject)
    db2_b_result = run_benchmark(base_path + "DB2_B", b_subjects, impressions_per_subject)
    db3_a_result = run_benchmark(base_path + "DB3_A", a_subjects, impressions_per_subject)
    db3_b_result = run_benchmark(base_path + "DB3_B", b_subjects, impressions_per_subject)
    db4_a_result = run_benchmark(base_path + "DB4_A", a_subjects, impressions_per_subject)
    db4_b_result = run_benchmark(base_path + "DB4_B", b_subjects, impressions_per_subject)

    # Show ERR
    print(f"db1_a_results: EER: {db1_a_result.get_equal_error_rate()}")
    print(f"db1_b_results: EER: {db1_b_result.get_equal_error_rate()}")
    print(f"db2_a_results: EER: {db2_a_result.get_equal_error_rate()}")
    print(f"db2_b_results: EER: {db2_b_result.get_equal_error_rate()}")
    print(f"db3_a_results: EER: {db3_a_result.get_equal_error_rate()}")
    print(f"db3_b_results: EER: {db3_b_result.get_equal_error_rate()}")
    print(f"db4_a_results: EER: {db4_a_result.get_equal_error_rate()}")
    print(f"db4_b_results: EER: {db4_b_result.get_equal_error_rate()}")

    thresholds = [.5, .55, .6, .65, .7, .75, .8, .85, .9, .95]
    db1_a_fmrs = np.array(db1_a_result.false_match_rate(thresholds))
    db1_a_fnmrs = np.array(db1_a_result.false_non_match_rate(thresholds))
    db1_b_fmrs = np.array(db1_b_result.false_match_rate(thresholds))
    db1_b_fnmrs = np.array(db1_b_result.false_match_rate(thresholds))
    db2_a_fmrs = np.array(db2_a_result.false_match_rate(thresholds))
    db2_a_fnmrs = np.array(db2_a_result.false_non_match_rate(thresholds))
    db2_b_fmrs = np.array(db2_b_result.false_match_rate(thresholds))
    db2_b_fnmrs = np.array(db2_b_result.false_match_rate(thresholds))
    db3_a_fmrs = np.array(db3_a_result.false_match_rate(thresholds))
    db3_a_fnmrs = np.array(db3_a_result.false_non_match_rate(thresholds))
    db3_b_fmrs = np.array(db3_b_result.false_match_rate(thresholds))
    db3_b_fnmrs = np.array(db3_b_result.false_match_rate(thresholds))
    db4_a_fmrs = np.array(db4_a_result.false_match_rate(thresholds))
    db4_a_fnmrs = np.array(db4_a_result.false_non_match_rate(thresholds))
    db4_b_fmrs = np.array(db4_b_result.false_match_rate(thresholds))
    db4_b_fnmrs = np.array(db4_b_result.false_match_rate(thresholds))

    generate_graph(thresholds, db1_a_fmrs, "FMR", "db1_a_fmrs")
    generate_graph(thresholds, db1_a_fnmrs, "FNMR", "db1_a_fnmrs")
    generate_graph(thresholds, db1_b_fmrs, "FMR", "db1_b_fmrs")
    generate_graph(thresholds, db1_b_fnmrs, "FNMR", "db1_b_fnmrs")
    generate_graph(thresholds, db2_a_fmrs, "FMR", "db2_a_fmrs")
    generate_graph(thresholds, db2_a_fnmrs, "FNMR", "db2_a_fnmrs")
    generate_graph(thresholds, db2_b_fmrs, "FMR", "db2_b_fmrs")
    generate_graph(thresholds, db2_b_fnmrs, "FNMR", "db2_b_fnmrs")
    generate_graph(thresholds, db3_a_fmrs, "FMR", "db3_a_fmrs")
    generate_graph(thresholds, db3_a_fnmrs, "FNMR", "db3_a_fnmrs")
    generate_graph(thresholds, db3_b_fmrs, "FMR", "db3_b_fmrs")
    generate_graph(thresholds, db3_b_fnmrs, "FNMR", "db3_b_fnmrs")
    generate_graph(thresholds, db4_a_fmrs, "FMR", "db4_a_fmrs")
    generate_graph(thresholds, db4_a_fnmrs, "FNMR", "db4_a_fnmrs")
    generate_graph(thresholds, db4_b_fmrs, "FMR", "db4_b_fmrs")
    generate_graph(thresholds, db4_b_fnmrs, "FNMR", "db4_b_fnmrs")


if __name__ == '__main__':
    freeze_support()
    main()
