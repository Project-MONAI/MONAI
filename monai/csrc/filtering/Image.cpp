#include "macros.h"
#include "Image.h"

// The rest of the Image class is inlined.
// Inlining the destructor makes the compiler unhappy, so it goes here instead

Image::~Image() {    
    if (!refCount) {
        return; // the image was a dummy
    }

    refCount[0]--;
    if (*refCount <= 0) {
        delete refCount;
        delete[] data;
    }
}
    
