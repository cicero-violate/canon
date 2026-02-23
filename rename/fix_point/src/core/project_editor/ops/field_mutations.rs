use anyhow::Result;


use quote::ToTokens;


use crate::state::NodeHandle;


use crate::structured::FieldMutation;


use super::helpers::{rename_ident_in_item, resolve_target_mut, TargetItemMut};
