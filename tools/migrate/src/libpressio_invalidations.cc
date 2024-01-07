#include "libpressio_modernize.h"
#include <fmt/format.h>
#include <fmt/ranges.h>
namespace clang {
namespace tidy {
namespace libpressio {


using namespace clang::ast_matchers;
void LibPressioAddInvalidationsCheck::registerMatchers(ast_matchers::MatchFinder *Finder) {
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
    Finder->addMatcher(
            traverse(TK_IgnoreUnlessSpelledInSource,
            cxxMethodDecl(
                hasName("set_options_impl"),
                forEachDescendant(
                    cxxMemberCallExpr(
                        hasArgument(4, memberExpr().bind("meta_many_child")), 
                        hasDeclaration(cxxMethodDecl(hasName("get_meta_many")))
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
                )), this);
    Finder->addMatcher(
            traverse(TK_IgnoreUnlessSpelledInSource,
            cxxMethodDecl(
                hasName("set_options_impl"),
                forEachDescendant(
                    cxxMemberCallExpr(
                        hasArgument(4, memberExpr().bind("meta_child")), 
                        hasDeclaration(cxxMethodDecl(hasName("get_meta")))
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
                )), this);
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
void LibPressioAddInvalidationsCheck::check(const ast_matchers::MatchFinder::MatchResult &Result) {
    if(const auto* option = Result.Nodes.getNodeAs<StringLiteral>("option")) {
        options.emplace_back(option->getString());
    } else if (const auto* return_stmt = Result.Nodes.getNodeAs<ReturnStmt>("return")) {
        auto const* return_var_expr = Result.Nodes.getNodeAs<DeclRefExpr>("returnvar");
        return_var = return_var_expr->getFoundDecl()->getNameAsString();
        insertion_point = return_stmt;
    } else if (const auto* get_meta = Result.Nodes.getNodeAs<MemberExpr>("meta_child")) {
        metas.emplace_back(get_meta->getFoundDecl()->getNameAsString());
    } else if (const auto* get_meta_many = Result.Nodes.getNodeAs<MemberExpr>("meta_many_child")) {
        many_metas.emplace_back(get_meta_many->getFoundDecl()->getNameAsString());
    } else if (const auto* config_option = Result.Nodes.getNodeAs<StringLiteral>("config_option")) {
        const auto str = config_option->getString();
        if (str == "predictors:error_dependent" ||
            str == "predictors:error_agnostic" ||
            str == "predictors:runtime") {
            skip = true;
        }
    }
}

void LibPressioAddInvalidationsCheck::onEndOfTranslationUnit() {
    constexpr auto invalidations_template = R"(
        //TODO fix the list of options for each command
        std::vector<std::string> invalidations {{"{1}"}}; 
        std::vector<pressio_configurable const*> invalidation_children {{{2}}}; 
        {3}
        set({0}, "predictors:error_dependent", get_accumulate_configuration("predictors:error_dependent", invalidation_children, invalidations));
        set({0}, "predictors:error_agnostic", get_accumulate_configuration("predictors:error_agnostic", invalidation_children, invalidations));
        set({0}, "predictors:runtime", get_accumulate_configuration("predictors:runtime", invalidation_children, invalidations));

    )";
    if(!skip && insertion_point != nullptr) {
        auto const formatted_metas = [this]{
            std::vector<std::string> f;
            f.reserve(metas.size());
            for (auto const& i : metas) {
                f.emplace_back(fmt::format("&*{}", i));
            }
            return f;
        }();
        auto const formatted_many_metas = [this]{
            std::vector<std::string> f;
            f.reserve(many_metas.size());
            for (auto const& i : many_metas) {
                f.emplace_back(fmt::format(R"(
            invalidation_children.reserve(invalidation_children.size() + {0}.size());
            for (auto const& child : {0}) {{
                invalidation_children.emplace_back(&*child);
            }}
                )", i));
            }
            return f;
        }();

        auto const fixithint = fmt::format(invalidations_template,
                return_var,
                fmt::join(options, "\", \""),
                fmt::join(formatted_metas, ", "),
                fmt::join(formatted_many_metas, "\n")
                );
        diag(insertion_point->getBeginLoc(), "provide invaliations to help tools determine when to re-run metrics")
            << FixItHint::CreateInsertion(insertion_point->getBeginLoc(), fixithint);
    }

    return_var.clear();
    metas.clear();
    many_metas.clear();
    insertion_point = nullptr;
    skip=false;
}

}}}
