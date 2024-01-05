#include "libpressio_modernize.h"
namespace clang {
namespace tidy {
namespace libpressio {

static ClangTidyModuleRegistry::Add<LibPressioModule> X("libpressio",
                                                "LibPressioChecks");

}}}
