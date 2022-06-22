
extern "C" {
void static_deleter(void*, void*) {
}
void static_copy(void** dest_ptr, void**, const void* src_ptr, const void*) {
  *dest_ptr = (void*)src_ptr;
}
}
