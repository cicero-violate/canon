| Item                    | Import path                                        |
| ----------------------- | -------------------------------------------------- |
| Crate name              | `use rustc_session::Session;`                      |
| Crate hash              | `use rustc_session::Session;`                      |
| Crate disambiguator     | `use rustc_session::Session;`                      |
| Crate edition           | `use rustc_session::config::Edition;`              |
| Crate source path       | `use rustc_session::Session;`                      |
| Crate dependencies      | `use rustc_metadata::creader::CStore;`             |
| Crate features          | `use rustc_session::Session;`                      |
| Crate target triple     | `use rustc_session::Session;`                      |
| Crate cfg flags         | `use rustc_session::Session;`                      |
| Module DefId            | `use rustc_hir::def_id::DefId;`                    |
| Module path             | `use rustc_middle::ty::TyCtxt;`                    |
| Module visibility       | `use rustc_middle::ty::Visibility;`                |
| Module attributes       | `use rustc_hir::Attribute;`                        |
| Item DefId              | `use rustc_hir::def_id::DefId;`                    |
| Item kind               | `use rustc_hir::ItemKind;`                         |
| Item name               | `use rustc_middle::ty::TyCtxt;`                    |
| Item path               | `use rustc_middle::ty::TyCtxt;`                    |
| Item visibility         | `use rustc_middle::ty::Visibility;`                |
| Item span               | `use rustc_span::Span;`                            |
| Item attributes         | `use rustc_hir::Attribute;`                        |
| Function DefId          | `use rustc_hir::def_id::DefId;`                    |
| Function name           | `use rustc_middle::ty::TyCtxt;`                    |
| Function signature      | `use rustc_middle::ty::FnSig;`                     |
| Function ABI            | `use rustc_middle::ty::FnSig;`                     |
| Function generics       | `use rustc_middle::ty::Generics;`                  |
| Function predicates     | `use rustc_middle::ty::GenericPredicates;`         |
| Function visibility     | `use rustc_middle::ty::Visibility;`                |
| Function attributes     | `use rustc_hir::Attribute;`                        |
| Function body HIR       | `use rustc_hir::Body;`                             |
| Function MIR            | `use rustc_middle::mir::Body;`                     |
| Function locals         | `use rustc_middle::mir::Local;`                    |
| Function return type    | `use rustc_middle::ty::Ty;`                        |
| Function constness      | `use rustc_hir::Constness;`                        |
| Function asyncness      | `use rustc_hir::IsAsync;`                          |
| Function unsafety       | `use rustc_hir::Unsafety;`                         |
| Struct DefId            | `use rustc_hir::def_id::DefId;`                    |
| Struct name             | `use rustc_middle::ty::TyCtxt;`                    |
| Struct fields           | `use rustc_hir::FieldDef;`                         |
| Struct field types      | `use rustc_middle::ty::Ty;`                        |
| Struct field visibility | `use rustc_middle::ty::Visibility;`                |
| Struct repr attributes  | `use rustc_middle::ty::ReprOptions;`               |
| Struct generics         | `use rustc_middle::ty::Generics;`                  |
| Enum DefId              | `use rustc_hir::def_id::DefId;`                    |
| Enum variants           | `use rustc_hir::Variant;`                          |
| Enum discriminants      | `use rustc_middle::ty::AdtDef;`                    |
| Union fields            | `use rustc_hir::FieldDef;`                         |
| Trait DefId             | `use rustc_hir::def_id::DefId;`                    |
| Trait methods           | `use rustc_middle::ty::AssocItems;`                |
| Trait associated types  | `use rustc_middle::ty::AssocItem;`                 |
| Trait supertraits       | `use rustc_middle::ty::TraitDef;`                  |
| Trait generics          | `use rustc_middle::ty::Generics;`                  |
| Impl DefId              | `use rustc_hir::def_id::DefId;`                    |
| Impl trait ref          | `use rustc_middle::ty::TraitRef;`                  |
| Impl methods            | `use rustc_middle::ty::AssocItems;`                |
| Impl generics           | `use rustc_middle::ty::Generics;`                  |
| Type alias DefId        | `use rustc_hir::def_id::DefId;`                    |
| Alias type              | `use rustc_middle::ty::Ty;`                        |
| Const DefId             | `use rustc_hir::def_id::DefId;`                    |
| Const value             | `use rustc_middle::mir::Const;`                    |
| Static DefId            | `use rustc_hir::def_id::DefId;`                    |
| Static initializer      | `use rustc_middle::mir::Body;`                     |
| Macro DefId             | `use rustc_hir::def_id::DefId;`                    |
| Macro kind              | `use rustc_hir::MacroKind;`                        |
| Macro span              | `use rustc_span::Span;`                            |
| HIR body id             | `use rustc_hir::BodyId;`                           |
| HIR expressions         | `use rustc_hir::Expr;`                             |
| HIR statements          | `use rustc_hir::Stmt;`                             |
| HIR patterns            | `use rustc_hir::Pat;`                              |
| MIR basic blocks        | `use rustc_middle::mir::BasicBlock;`               |
| MIR statements          | `use rustc_middle::mir::Statement;`                |
| MIR terminators         | `use rustc_middle::mir::Terminator;`               |
| MIR locals              | `use rustc_middle::mir::Local;`                    |
| MIR local types         | `use rustc_middle::mir::LocalDecl;`                |
| MIR source scopes       | `use rustc_middle::mir::SourceScope;`              |
| MIR spans               | `use rustc_span::Span;`                            |
| Call terminators        | `use rustc_middle::mir::TerminatorKind;`           |
| DefId → symbol string   | `use rustc_middle::ty::TyCtxt;`                    |
| DefId → path string     | `use rustc_middle::ty::TyCtxt;`                    |
| Type of DefId           | `use rustc_middle::ty::TyCtxt;`                    |
| Type layout             | `use rustc_middle::ty::layout::TyAndLayout;`       |
| Borrowck facts          | `use rustc_middle::mir::borrowck;`                 |
| Move paths              | `use rustc_middle::mir::MovePath;`                 |
| Loans                   | `use rustc_middle::mir::borrowck;`                 |
| Optimized MIR           | `use rustc_middle::ty::TyCtxt;`                    |
| Doc comments            | `use rustc_hir::Attribute;`                        |
| Span → file             | `use rustc_span::source_map::SourceMap;`           |
| Source file contents    | `use rustc_span::source_map::SourceMap;`           |
| Crate-level lang items  | `use rustc_middle::middle::lang_items::LangItems;` |
| Target data layout      | `use rustc_target::abi::TargetDataLayout;`         |
| Extern crates           | `use rustc_metadata::creader::CStore;`             |
| Feature gates           | `use rustc_session::Session;`                      |
| Lint levels             | `use rustc_lint::LintStore;`                       |
| Macro expansions        | `use rustc_expand::base::ExtCtxt;`                 |
| Monomorphized instances | `use rustc_middle::ty::Instance;`                  |
| VTable entries          | `use rustc_middle::ty::vtable::VtblEntry;`         |
| Drop glue instances     | `use rustc_middle::ty::Instance;`                  |
| Associated item mapping | `use rustc_middle::ty::AssocItems;`                |
| Module tree edges       | `use rustc_hir::Mod;`                              |
| Item ownership edges    | `use rustc_hir::OwnerId;`                          |

