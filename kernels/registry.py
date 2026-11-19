"""The kernels the compiler maps, by model name: get_kernel(model)."""

KNOWN_MODELS = (
    "vit-base", "vit-large", "bert-base", "bert-base-ffn-smooth", "bert-large", "gpt2",
    "resnet50", "mobilenet-v2", "vgg16",
    "bert-attention-12layer", "bert-attention-1layer",
    # group-wise 3-bit weights (kernels/w3.py): the RRAM/SRAM ensemble's format
    "resnet50-w3", "mobilenet-v2-w3", "vgg16-w3", "vit-base-w3", "vit-large-w3",
    "bert-base-w3", "bert-large-w3", "gpt2-w3", "bert-attention-12layer-w3",
)
# kernel of app <app> in the w3 format
W3_SUFFIX = "-w3"


def get_kernel(model: str):
    """Returns (KernelClass, INPUT_SHAPE, BATCH_SIZE) for `model`.

    INPUT_SHAPE is the per-sample input shape (everything except the batch dim): e.g.
    (MAX_SEQ_LEN, DIM) for a sequence model, (3, IMAGE_SIZE, IMAGE_SIZE) for an image
    model. Callers build the example input as torch.randn(batch_size, *INPUT_SHAPE).

    Imports are local to each branch so only that model's kernel module needs to import.
    """
    if model == "vit-base":
        from kernels.vit_base import ViTBaseEncoderInt8, DIM, MAX_SEQ_LEN, BATCH_SIZE
        return ViTBaseEncoderInt8, (MAX_SEQ_LEN, DIM), BATCH_SIZE
    if model == "vit-base-w3":
        from kernels.vit_base import ViTBaseEncoderW3, DIM, MAX_SEQ_LEN, BATCH_SIZE
        return ViTBaseEncoderW3, (MAX_SEQ_LEN, DIM), BATCH_SIZE
    if model == "vit-large":
        from kernels.vit import ViTEncoderInt8, DIM, MAX_SEQ_LEN, BATCH_SIZE
        return ViTEncoderInt8, (MAX_SEQ_LEN, DIM), BATCH_SIZE
    if model == "vit-large-w3":
        from kernels.vit import ViTEncoderW3, DIM, MAX_SEQ_LEN, BATCH_SIZE
        return ViTEncoderW3, (MAX_SEQ_LEN, DIM), BATCH_SIZE
    if model == "bert-base":
        from kernels.bert_base import BertBaseKernelInt8, DIM, MAX_SEQ_LEN, BATCH_SIZE
        return BertBaseKernelInt8, (MAX_SEQ_LEN, DIM), BATCH_SIZE
    if model == "bert-base-ffn-smooth":
        # SmoothQuant on fc1 and fc2 only (no per-channel x / s before Wq / Wk / Wv)
        from kernels.bert_base import BertBaseKernelInt8FfnSmooth, DIM, MAX_SEQ_LEN, BATCH_SIZE
        return BertBaseKernelInt8FfnSmooth, (MAX_SEQ_LEN, DIM), BATCH_SIZE
    if model == "bert-base-w3":
        from kernels.bert_base import BertBaseKernelW3, DIM, MAX_SEQ_LEN, BATCH_SIZE
        return BertBaseKernelW3, (MAX_SEQ_LEN, DIM), BATCH_SIZE
    if model == "bert-large":
        from kernels.bert_large import BertLargeKernelInt8, DIM, MAX_SEQ_LEN, BATCH_SIZE
        return BertLargeKernelInt8, (MAX_SEQ_LEN, DIM), BATCH_SIZE
    if model == "bert-large-w3":
        from kernels.bert_large import BertLargeKernelW3, DIM, MAX_SEQ_LEN, BATCH_SIZE
        return BertLargeKernelW3, (MAX_SEQ_LEN, DIM), BATCH_SIZE
    if model == "gpt2":
        from kernels.gpt2 import GPT2KernelInt8, DIM, MAX_SEQ_LEN, BATCH_SIZE
        return GPT2KernelInt8, (MAX_SEQ_LEN, DIM), BATCH_SIZE
    if model == "gpt2-w3":
        # w3 weights on the per-token path
        from kernels.gpt2 import GPT2KernelW3, DIM, MAX_SEQ_LEN, BATCH_SIZE
        return GPT2KernelW3, (MAX_SEQ_LEN, DIM), BATCH_SIZE
    if model == "resnet50":
        from kernels.resnet50 import ResNet50Int8, INPUT_SHAPE, BATCH_SIZE
        return ResNet50Int8, INPUT_SHAPE, BATCH_SIZE
    if model == "resnet50-w3":
        from kernels.resnet50 import ResNet50W3, INPUT_SHAPE, BATCH_SIZE
        return ResNet50W3, INPUT_SHAPE, BATCH_SIZE
    if model == "mobilenet-v2":
        from kernels.mobilenet_v2 import MobileNetV2Int8, INPUT_SHAPE, BATCH_SIZE
        return MobileNetV2Int8, INPUT_SHAPE, BATCH_SIZE
    if model == "mobilenet-v2-w3":
        from kernels.mobilenet_v2 import MobileNetV2W3, INPUT_SHAPE, BATCH_SIZE
        return MobileNetV2W3, INPUT_SHAPE, BATCH_SIZE
    if model == "vgg16":
        from kernels.vgg16 import VGG16Int8, INPUT_SHAPE, BATCH_SIZE
        return VGG16Int8, INPUT_SHAPE, BATCH_SIZE
    if model == "vgg16-w3":
        from kernels.vgg16 import VGG16W3, INPUT_SHAPE, BATCH_SIZE
        return VGG16W3, INPUT_SHAPE, BATCH_SIZE
    if model == "bert-attention-12layer":
        # the int8 app (run_scripts/measure_accuracy/build_bert_attention_12layer_int8.py)
        from kernels.bert_attention_only import BertAttention12LayerKernelInt8, DIM, MAX_SEQ_LEN, BATCH_SIZE
        return BertAttention12LayerKernelInt8, (MAX_SEQ_LEN, DIM), BATCH_SIZE
    if model == "bert-attention-1layer":
        # its first layer alone (build_bert_attention_12layer_int8.py --layers 1), the paper's Fig. 6
        from kernels.bert_attention_only import BertAttention1LayerKernelInt8, DIM, MAX_SEQ_LEN, BATCH_SIZE
        return BertAttention1LayerKernelInt8, (MAX_SEQ_LEN, DIM), BATCH_SIZE
    if model == "bert-attention-12layer-w3":
        from kernels.bert_attention_only import BertAttention12LayerKernelW3, DIM, MAX_SEQ_LEN, BATCH_SIZE
        return BertAttention12LayerKernelW3, (MAX_SEQ_LEN, DIM), BATCH_SIZE
    raise ValueError(f"unknown model: {model!r}")
