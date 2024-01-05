#include "libpressio_modernize.h"
#include <fmt/format.h>
#include <fmt/ranges.h>
namespace clang {
namespace tidy {
namespace libpressio {


using namespace clang::ast_matchers;
void LibPressioAddHighLevelCheck::registerMatchers(ast_matchers::MatchFinder *Finder) {
    Finder->addMatcher(cxxMethodDecl(
                hasName("get_configuration_impl"),
                forEachDescendant(
                    cxxMemberCallExpr(
                        hasArgument(1, stringLiteral().bind("config_option")), 
                        hasDeclaration(cxxMethodDecl(hasName("set")))
                        )
                    ),
                hasAncestor(
                    cxxRecordDecl(
                        hasAnyBase(
                            hasType(
                                static_cast<ast_matchers::internal::Matcher<QualType>>(hasDeclaration(
                                    cxxRecordDecl(
                                        hasName("libpressio_compressor_plugin")
                                    )
                                )
                            ))
                        )
                    )
                )
                ), this);
    Finder->addMatcher(cxxMethodDecl(
                hasName("set_options_impl"),
                forEachDescendant(
                    cxxMemberCallExpr(
                        hasArgument(1, stringLiteral().bind("option")), 
                        hasDeclaration(cxxMethodDecl(hasName("get")))
                        )
                    ),
                hasAncestor(
                    cxxRecordDecl(
                        hasAnyBase(
                            hasType(
                                static_cast<ast_matchers::internal::Matcher<QualType>>(hasDeclaration(
                                    cxxRecordDecl(
                                        hasName("libpressio_compressor_plugin")
                                    )
                                )
                            ))
                        )
                    )
                )
                ), this);
    Finder->addMatcher(
            traverse(TK_IgnoreUnlessSpelledInSource,
            declRefExpr(
                hasParent(
                    returnStmt(
                        hasAncestor(
                            cxxMethodDecl(
                                hasName("get_configuration_impl")
                            )
                        )
                    ).bind("return")
                )).bind("returnvar")),
            this
            );
}
void LibPressioAddHighLevelCheck::check(const ast_matchers::MatchFinder::MatchResult &Result) {
    if(const auto* option = Result.Nodes.getNodeAs<StringLiteral>("option")) {
        auto option_str = option->getString();
        if(option_str.starts_with("pressio:")) {
            options.emplace_back(option_str);
        }
    } else if (const auto* return_stmt = Result.Nodes.getNodeAs<ReturnStmt>("return")) {
        auto const* return_var_expr = Result.Nodes.getNodeAs<DeclRefExpr>("returnvar");
        return_var = return_var_expr->getFoundDecl()->getNameAsString();
        insertion_point = return_stmt;
    } else if (const auto* config_option = Result.Nodes.getNodeAs<StringLiteral>("config_option")) {
        const auto str = config_option->getString();
        if (str == "pressio:highlevel") {
            skip = true;
        }
    }
}

void LibPressioAddHighLevelCheck::onEndOfTranslationUnit() {
    constexpr auto invalidations_template = R"(
        set({0}, "pressio:highlevel", get_accumulate_configuration("pressio:highlevel", invalidation_children, std::vector<std::string>{{"{1}"}}));

    )";

    if(!skip && insertion_point != nullptr && options.size() > 0) {
        auto const fixithint = fmt::format(invalidations_template,
                return_var,
                fmt::join(options, "\", \"")
                );
        diag(insertion_point->getBeginLoc(), "provide at least the pressio: keyed options as high level options")
            << FixItHint::CreateInsertion(insertion_point->getBeginLoc(), fixithint);
    }

    return_var.clear();
    insertion_point = nullptr;
    skip=false;
}

}}}
