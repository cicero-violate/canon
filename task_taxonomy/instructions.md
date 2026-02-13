### File Operations
| Action       | Params          |
|--------------+-----------------|
| `CreateFile` | file, template? |
| `DeleteFile` | file            |
| `RenameFile` | file, new_name  |
| `MoveFile`   | file, new_path  |

### Struct / Enum / Trait Shape
| Action              | Params                                                 |
|---------------------+--------------------------------------------------------|
| `AddField`          | file, struct_name, field_name, field_type, visibility? |
| `RemoveField`       | file, struct_name, field_name                          |
| `RenameField`       | file, struct_name, old_name, new_name                  |
| `AddEnumVariant`    | file, enum_name, variant, payload_type?                |
| `RemoveEnumVariant` | file, enum_name, variant                               |
| `RenameEnumVariant` | file, enum_name, old_name, new_name                    |
| `AddTraitMethod`    | file, trait_name, signature                            |
| `RemoveTraitMethod` | file, trait_name, method_name                          |

### Imports / Exports
| Action          | Params                      |
|-----------------+-----------------------------|
| `AddUseDecl`    | file, use_path              |
| `RemoveUseDecl` | file, use_path              |
| `AddModDecl`    | file, mod_name, visibility? |
| `AddPubExport`  | file, symbol_name           |

### Attributes / Derives
| Action            | Params                       |
|-------------------+------------------------------|
| `AddDerive`       | file, type_name, trait_name  |
| `RemoveDerive`    | file, type_name, trait_name  |
| `AddAttribute`    | file, symbol_name, attribute |
| `RemoveAttribute` | file, symbol_name, attribute |

### Cargo
| Action             | Params                           |
|--------------------+----------------------------------|
| `AddDependency`    | crate_name, version, features[]? |
| `RemoveDependency` | crate_name                       |
| `AddFeatureFlag`   | feature_name, deps[]?            |
| `SetEdition`       | edition                          |

---

### Write New Code
| Action           | Params                                  |
|------------------+-----------------------------------------|
| `WriteFunction`  | file, signature, context_symbols[]      |
| `WriteStruct`    | file, struct_name, fields[], derives[]? |
| `WriteEnum`      | file, enum_name, variants[], derives[]? |
| `WriteTrait`     | file, trait_name, method_signatures[]   |
| `WriteImplBlock` | file, type_name, trait_name?, methods[] |
| `WriteTypeAlias` | file, alias_name, target_type           |
| `WriteConstant`  | file, name, type, description           |
| `WriteModule`    | file, mod_name, description             |
| `WriteMacro`     | file, macro_name, description           |

### Extend Existing Code
| Action            | Params                                         |
|-------------------+------------------------------------------------|
| `AddMethodToImpl` | file, type_name, signature, context_symbols[]  |
| `AddMatchArm`     | file, fn_name, variant, handler_description    |
| `AddBranch`       | file, fn_name, condition_description           |
| `ImplementTrait`  | file, type_name, trait_name, context_symbols[] |

### Boilerplate Generation
| Action           | Params                       |
|------------------+------------------------------|
| `WriteBuilder`   | file, struct_name            |
| `WriteDefault`   | file, type_name              |
| `WriteDisplay`   | file, type_name              |
| `WriteFromInto`  | file, from_type, into_type   |
| `WriteSerdeImpl` | file, type_name              |
| `WriteIterator`  | file, type_name, item_type   |
| `WriteError`     | file, error_name, variants[] |

---

### Repair
| Action             | Params                   |
|--------------------+--------------------------|
| `FixCompileError`  | file, error_text, line   |
| `FixBorrowError`   | file, error_text, line   |
| `FixTypeError`     | file, error_text, line   |
| `FixLifetimeError` | file, error_text, line   |
| `FixUnusedWarning` | file, warning_text, line |
| `FixClippyLint`    | file, lint_name, line    |

### Refactor
| Action             | Params                                 |
|--------------------+----------------------------------------|
| `RefactorFunction` | file, fn_name, instruction             |
| `RefactorStruct`   | file, struct_name, instruction         |
| `ExtractFunction`  | file, fn_name, line_range, new_fn_name |
| `InlineFunction`   | file, fn_name, call_site               |
| `SplitModule`      | file, new_files[], split_description   |
| `MergeModules`     | files[], target_file                   |
| `RenameSymbol`     | file, old_name, new_name, scope        |

### Migration
| Action                | Params                                                 |
|-----------------------+--------------------------------------------------------|
| `MigrateToTrait`      | file, type_name, trait_name                            |
| `MigrateErrorType`    | file, old_error, new_error                             |
| `MigrateAsyncRuntime` | file, from_runtime, to_runtime                         |
| `UpgradeDependency`   | crate_name, old_version, new_version, breaking_changes |

---

## Layer 3 — Meta / Control Flow (no code output, orchestration only)

These are not edits. They are signals that change the execution graph itself.

| Action                 | Params                                                        |
|------------------------+---------------------------------------------------------------|
| `CargoCheck`           | — captures errors → spawns FixCompileError children           |
| `CargoClippy`          | — captures lints → spawns FixClippyLint children              |
| `RebuildRepomap`       | — re-runs tree-sitter scan after edits, updates R             |
| `SplitTask`            | task_id, reason → replanner call on one task                  |
| `RetryTask`            | task_id, prior_error → retry with error injected into context |
| `AbandonBranch`        | task_id → mark tlog branch dead, time-travel to parent        |
| `RequestClarification` | question → pause execution, surface to user                   |
| `PlanSubgoal`          | description → recursive planner call, spawns child DAG        |
