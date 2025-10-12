from monai.data.image_writer import resolve_writer, OptionalImportError, SUPPORTED_WRITERS, EXT_WILDCARD, ITKWriter

# Fake extension to simulate unsupported file type
ext = "fakeext"

print(f"Before clearing fallback writers: {SUPPORTED_WRITERS.get(EXT_WILDCARD)}")

# Temporarily clear fallback writers to simulate no support scenario
SUPPORTED_WRITERS[EXT_WILDCARD] = ()

try:
    # Try resolving writer for fake unsupported extension with error flag
    writers = resolve_writer(ext, error_if_not_found=True)
except OptionalImportError as e:
    # Catch and print the enhanced OptionalImportError with package hints
    print("Caught OptionalImportError:", e)
finally:
    # Restore the fallback writers to avoid side effects on other tests
    SUPPORTED_WRITERS[EXT_WILDCARD] = (ITKWriter,)

print(f"After restoring fallback writers: {SUPPORTED_WRITERS.get(EXT_WILDCARD)}")
