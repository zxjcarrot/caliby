#ifndef ALIGNED_ALLOCATOR_HPP
#define ALIGNED_ALLOCATOR_HPP

#include <cstddef>
#include <cstdlib>
#include <memory>
#include <new>

// Custom allocator for std::vector to ensure aligned memory allocation,
// crucial for SIMD performance.
template <typename T, size_t Alignment>
class AlignedAllocator {
public:
    using value_type = T;
    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using const_reference = const T&;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

    template <typename U>
    struct rebind {
        using other = AlignedAllocator<U, Alignment>;
    };

    AlignedAllocator() noexcept {}

    template <typename U>
    AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}

    pointer address(reference x) const noexcept {
        return std::addressof(x);
    }

    const_pointer address(const_reference x) const noexcept {
        return std::addressof(x);
    }

    pointer allocate(size_type n) {
        if (n == 0) {
            return nullptr;
        }
        if (n > std::numeric_limits<size_type>::max() / sizeof(T)) {
            throw std::bad_alloc();
        }
        void* p;
#if defined(_MSC_VER)
        p = _aligned_malloc(n * sizeof(T), Alignment);
        if (!p) {
            throw std::bad_alloc();
        }
#else
        if (posix_memalign(&p, Alignment, n * sizeof(T)) != 0) {
            throw std::bad_alloc();
        }
#endif
        return static_cast<pointer>(p);
    }

    void deallocate(pointer p, size_type) noexcept {
#if defined(_MSC_VER)
        _aligned_free(p);
#else
        free(p);
#endif
    }

    size_type max_size() const noexcept {
        return std::numeric_limits<size_type>::max() / sizeof(T);
    }
};

template <typename T, size_t TAlignment, typename U, size_t UAlignment>
inline bool operator==(const AlignedAllocator<T, TAlignment>&, const AlignedAllocator<U, UAlignment>&) noexcept {
    return TAlignment == UAlignment;
}

template <typename T, size_t TAlignment, typename U, size_t UAlignment>
inline bool operator!=(const AlignedAllocator<T, TAlignment>&, const AlignedAllocator<U, UAlignment>&) noexcept {
    return TAlignment != UAlignment;
}

#endif // ALIGNED_ALLOCATOR_HPP