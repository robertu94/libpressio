#include <clang-tidy/ClangTidyCheck.h>
#include <clang-tidy/ClangTidyModule.h>
#include <clang-tidy/ClangTidyModuleRegistry.h>

namespace clang {
namespace tidy {
namespace libpressio {


class LibPressioAddHighLevelCheck : public ClangTidyCheck {
public:
  LibPressioAddHighLevelCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;

  void onEndOfTranslationUnit() override;

  std::vector<std::string> options;
  std::string return_var;
  ReturnStmt const* insertion_point;
  bool skip = false;
  bool has_invalidation_children = false;

};

class LibPressioAddInvalidationsCheck : public ClangTidyCheck {
public:
  LibPressioAddInvalidationsCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;

  void onEndOfTranslationUnit() override;

  std::vector<std::string> options;
  std::vector<std::string> metas;
  std::vector<std::string> many_metas;
  std::string return_var;
  ReturnStmt const* insertion_point;
  bool skip = false;

};

} // namespace readability

class LibPressioModule : public ClangTidyModule {
 public:
  void addCheckFactories(ClangTidyCheckFactories &CheckFactories) override {
    CheckFactories.registerCheck<libpressio::LibPressioAddInvalidationsCheck>(
        "libpressio_invalidation");
    CheckFactories.registerCheck<libpressio::LibPressioAddHighLevelCheck>(
        "libpressio_highlevel");
  }
};

} // namespace tidy
} // namespace clang
