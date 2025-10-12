from monai.data.image_writer import resolve_writer, OptionalImportError, SUPPORTED_WRITERS, EXT_WILDCARD, ITKWriter

ext = "fakeext"

print(f"Before clearing fallback writers: {SUPPORTED_WRITERS.get(EXT_WILDCARD)}")

# Temporarily clear fallback writers to simulate no support
SUPPORTED_WRITERS[EXT_WILDCARD] = ()

try:
    writers = resolve_writer(ext, error_if_not_found=True)
except OptionalImportError as e:
    print("Caught OptionalImportError:", e)
finally:
    # Restore the fallback writers to avoid side effects
    SUPPORTED_WRITERS[EXT_WILDCARD] = (ITKWriter,)

print(f"After restoring fallback writers: {SUPPORTED_WRITERS.get(EXT_WILDCARD)}")
