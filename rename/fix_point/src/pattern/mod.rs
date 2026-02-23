pub mod binding;

use crate::pattern::binding::PatternBindingCollector;


use syn::visit::Visit;


use syn::{Pat, PatIdent, PatSlice, PatStruct, PatTuple, PatTupleStruct};
