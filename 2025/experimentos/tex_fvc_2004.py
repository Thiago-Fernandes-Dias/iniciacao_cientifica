from flx.benchmarks.matchers import CosineSimilarityMatcher
from flx.data.dataset import Dataset, Identifier, IdentifierSet
from flx.data.image_loader import FVC2004Loader
from flx.data.label_index import LabelIndex
from flx.extractor.fixed_length_extractor import DeepPrintExtractor, get_DeepPrint_Tex
from flx.scripts.generate_benchmarks import create_verification_benchmark

NUM_IMPRESSIONS = 8


def main():
    training_ids = IdentifierSet(
        [Identifier(s, i) for s in range(100) for i in range(8)]
    )

    deep_print_tex_extractor: DeepPrintExtractor = get_DeepPrint_Tex(
        num_training_subjects=training_ids.num_subjects, num_texture_dims=256
    )

    training_dataset = Dataset(FVC2004Loader("D:\Fingerprints_dataset\FVC\FVC2004\Dbs\DB1_A"), training_ids)

    label_dataset = Dataset(LabelIndex(training_ids), training_ids)

    deep_print_tex_extractor.fit(
        fingerprints=training_dataset,
        minutia_maps=None,
        labels=label_dataset,
        validation_fingerprints=None,
        validation_benchmark=None,
        num_epochs=100,
        out_dir="./models",
    )

    deep_print_tex_extractor.load_best_model("./models")

    test_dataset = Dataset(
        FVC2004Loader("D:\Fingerprints_dataset\FVC\FVC2004\Dbs\DB1_B"),
        IdentifierSet([Identifier(s, i) for s in range(101, 110) for i in range(8)]),
    )

    tex_embeddings, _ = deep_print_tex_extractor.extract(test_dataset)

    benchmark = create_verification_benchmark(
        subjects=list(range(101, 110)),
        impressions_per_subject=list(range(6, 8)),
    )

    matcher = CosineSimilarityMatcher(tex_embeddings)

    results = benchmark.run(matcher)

    print(f"EER: {results.get_equal_error_rate()}")


if __name__ == "__main__":
    # freeze_support()
    main()
