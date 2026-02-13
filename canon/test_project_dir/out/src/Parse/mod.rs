// Derived from Canonical IR. Do not edit.

struct ParseState {
    // no fields
}

trait FilePath {
    fn Filepath() -> FilepathResult;
}

trait LintReport {
    fn Lintreport() -> LintreportResult;
}

trait Test2 {
    fn Test2() -> Test2Result;
}

trait Test3 {
    fn Test3() -> Test3Result;
}

impl FilePath for ParseState {

    fn FilepathParseState() -> FilepathResult {
    // Canon runtime stub
    canon_runtime::execute_function("fn.impl_struct_module_parse_state_trait_module_parse_filepath.trait_fn_module_parse_filepath");
    }

}

impl LintReport for ParseState {

    fn LintreportParseState() -> LintreportResult {
    // Canon runtime stub
    canon_runtime::execute_function("fn.impl_struct_module_parse_state_trait_module_parse_lintreport.trait_fn_module_parse_lintreport");
    }

}

impl Test2 for ParseState {

    fn Test2ParseState() -> Test2Result {
    // Canon runtime stub
    canon_runtime::execute_function("fn.impl_struct_module_parse_state_trait_module_parse_test2.trait_fn_module_parse_test2");
    }

}

impl Test3 for ParseState {

    fn Test3ParseState() -> Test3Result {
    // Canon runtime stub
    canon_runtime::execute_function("fn.impl_struct_module_parse_state_trait_module_parse_test3.trait_fn_module_parse_test3");
    }

}
