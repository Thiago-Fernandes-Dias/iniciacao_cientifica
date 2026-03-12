from flx.benchmarks.matchers import CosineSimilarityMatcher
from flx.data.dataset import Dataset
from flx.data.image_loader import FVC2004Loader
from flx.extractor.fixed_length_extractor import DeepPrintExtractor
from flx.scripts.generate_benchmarks import create_verification_benchmark

from flx.data.transformed_image_loader import TransformedImageLoader
from flx.image_processing.binarization import LazilyAllocatedBinarizer
from flx.data.image_helpers import pad_and_resize_to_deepprint_input_size

from flx.extractor.fixed_length_extractor import get_DeepPrint_TexMinu


def main():
    # modelo pré-treinado disponibilizado no repositório
    deep_print_tex_extractor: DeepPrintExtractor = get_DeepPrint_TexMinu(
        num_training_subjects=8000, num_dims=256
    )
    deep_print_tex_extractor.load_best_model("./models")

    image_loader = TransformedImageLoader(
        images=FVC2004Loader(r"FVC2004\Dbs\DB1_B"),
        poses=None,
        transforms=[
            LazilyAllocatedBinarizer(5.0),
            pad_and_resize_to_deepprint_input_size,
        ],
    )

    test_dataset: Dataset = Dataset(image_loader, image_loader.ids)
    tex_embeddings, minutia_embeddings = deep_print_tex_extractor.extract(test_dataset)

    benchmark = create_verification_benchmark(
        subjects=list(range(100, 110)),
        impressions_per_subject=list(range(0, 8)),
    )

    from flx.data.embedding_loader import EmbeddingLoader

    matcher = CosineSimilarityMatcher(
        EmbeddingLoader.combine(tex_embeddings, minutia_embeddings)
    )
    # matcher = CosineSimilarityMatcher(tex_embeddings) - tbm funciona, mas a EER aumentou um pouco nos testes que fiz

    results = benchmark.run(matcher)

    print(f"EER: {results.get_equal_error_rate()}")


if __name__ == "__main__":
    main()
