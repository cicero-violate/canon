use proc_macro2::Span;


use super::types::{LineColumn, SpanRange};


pub fn span_to_range(span: Span) -> SpanRange {
    let start = span.start();
    let end = span.end();
    SpanRange {
        start: LineColumn {
            line: start.line as i64,
            column: start.column as i64 + 1,
        },
        end: LineColumn {
            line: end.line as i64,
            column: end.column as i64 + 1,
        },
    }
}
