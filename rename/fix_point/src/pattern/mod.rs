pub mod binding;

pub fn extract_type_from_pattern(pat: &Pat) -> Option<String> {
    match pat {
        Pat::Type(pat_type) => Some(type_to_string(&pat_type.ty)),
        _ => None,
    }
}


pub fn extract_type_from_pattern(pat: &Pat) -> Option<String> {
    match pat {
        Pat::Type(pat_type) => Some(type_to_string(&pat_type.ty)),
        _ => None,
    }
}


fn type_to_string(ty: &syn::Type) -> String {
    match ty {
        syn::Type::Path(type_path) => path_to_string(&type_path.path),
        syn::Type::Reference(type_ref) => format!("&{}", type_to_string(& type_ref.elem)),
        syn::Type::Tuple(type_tuple) => {
            let elems: Vec<String> = type_tuple
                .elems
                .iter()
                .map(type_to_string)
                .collect();
            format!("({})", elems.join(", "))
        }
        syn::Type::Slice(type_slice) => {
            format!("[{}]", type_to_string(& type_slice.elem))
        }
        _ => "Unknown".to_string(),
    }
}


fn path_to_string(path: &syn::Path) -> String {
    path.segments.iter().map(|seg| seg.ident.to_string()).collect::<Vec<_>>().join("::")
}


fn type_to_string(ty: &syn::Type) -> String {
    match ty {
        syn::Type::Path(type_path) => path_to_string(&type_path.path),
        syn::Type::Reference(type_ref) => format!("&{}", type_to_string(& type_ref.elem)),
        syn::Type::Tuple(type_tuple) => {
            let elems: Vec<String> = type_tuple
                .elems
                .iter()
                .map(type_to_string)
                .collect();
            format!("({})", elems.join(", "))
        }
        syn::Type::Slice(type_slice) => {
            format!("[{}]", type_to_string(& type_slice.elem))
        }
        _ => "Unknown".to_string(),
    }
}


fn path_to_string(path: &syn::Path) -> String {
    path.segments.iter().map(|seg| seg.ident.to_string()).collect::<Vec<_>>().join("::")
}


pub fn extract_type_from_pattern(pat: &Pat) -> Option<String> {
    match pat {
        Pat::Type(pat_type) => Some(type_to_string(&pat_type.ty)),
        _ => None,
    }
}


pub fn extract_type_from_pattern(pat: &Pat) -> Option<String> {
    match pat {
        Pat::Type(pat_type) => Some(type_to_string(&pat_type.ty)),
        _ => None,
    }
}


fn type_to_string(ty: &syn::Type) -> String {
    match ty {
        syn::Type::Path(type_path) => path_to_string(&type_path.path),
        syn::Type::Reference(type_ref) => format!("&{}", type_to_string(& type_ref.elem)),
        syn::Type::Tuple(type_tuple) => {
            let elems: Vec<String> = type_tuple
                .elems
                .iter()
                .map(type_to_string)
                .collect();
            format!("({})", elems.join(", "))
        }
        syn::Type::Slice(type_slice) => {
            format!("[{}]", type_to_string(& type_slice.elem))
        }
        _ => "Unknown".to_string(),
    }
}


fn path_to_string(path: &syn::Path) -> String {
    path.segments.iter().map(|seg| seg.ident.to_string()).collect::<Vec<_>>().join("::")
}


fn type_to_string(ty: &syn::Type) -> String {
    match ty {
        syn::Type::Path(type_path) => path_to_string(&type_path.path),
        syn::Type::Reference(type_ref) => format!("&{}", type_to_string(& type_ref.elem)),
        syn::Type::Tuple(type_tuple) => {
            let elems: Vec<String> = type_tuple
                .elems
                .iter()
                .map(type_to_string)
                .collect();
            format!("({})", elems.join(", "))
        }
        syn::Type::Slice(type_slice) => {
            format!("[{}]", type_to_string(& type_slice.elem))
        }
        _ => "Unknown".to_string(),
    }
}


fn path_to_string(path: &syn::Path) -> String {
    path.segments.iter().map(|seg| seg.ident.to_string()).collect::<Vec<_>>().join("::")
}
