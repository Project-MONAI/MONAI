from monai.utils import OptionalImportError

__all__ = ["MonaiHubMixin"]


class DummyPyTorchModelHubMixin:

    error_message = "To use `{}` method please required packages: `pip install huggingface_hub safetensors`."

    def __init_subclass__(cls, *args, **kwargs):
        super().__init_subclass__()

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        raise OptionalImportError(cls.error_message.format("from_pretrained"))

    def save_pretrained(self, *args, **kwargs):
        raise OptionalImportError(self.error_message.format("save_pretrained"))

    def push_to_hub(self, *args, **kwargs):
        raise OptionalImportError(self.error_message.format("push_to_hub"))


try:
    from huggingface_hub import PyTorchModelHubMixin
except ImportError:
    PyTorchModelHubMixin = DummyPyTorchModelHubMixin


class MonaiHubMixin(
    PyTorchModelHubMixin,
    library_name="monai",
    repo_url="https://github.com/Project-MONAI/MONAI",
    docs_url="https://docs.monai.io/en/",
    tags=["monai"],
):
    pass
