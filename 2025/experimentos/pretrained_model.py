from multiprocessing.dummy import freeze_support
from flx.benchmarks.matchers import CosineSimilarityMatcher
from flx.data.dataset import Dataset, Identifier, IdentifierSet
from flx.data.image_loader import FVC2004Loader
from flx.extractor.fixed_length_extractor import get_DeepPrint_TexMinu
from flx.scripts.generate_benchmarks import create_verification_benchmark

def main():
    extractor = get_DeepPrint_TexMinu(num_training_subjects=8000, num_dims=256)
    extractor.load_best_model("./fixed-length-fingerprint-extractors/models")

    test_dataset = Dataset(
        FVC2004Loader("D:\Fingerprints_dataset\FVC\FVC2004\Dbs\DB1_A"),
        IdentifierSet([Identifier(s, i) for s in range(100) for i in range(8)]),
    )

    tex_embeddings, _ = extractor.extract(test_dataset)

    benchmark = create_verification_benchmark(
        subjects=list(range(100)),
        impressions_per_subject=list(range(8)),
    )

    matcher = CosineSimilarityMatcher(tex_embeddings)

    results = benchmark.run(matcher)

    print(f"EER: {results.get_equal_error_rate()}")
    print(f"FMR: {results.false_match_rate([.6, .7, .8, .9, .95, ])}")
    print(f"FNMR: {results.false_non_match_rate([.6, .7, .8, .9, .95])}")

if __name__ == "__main__":
    freeze_support()
    main()