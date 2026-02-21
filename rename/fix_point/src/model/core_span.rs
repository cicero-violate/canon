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


pub(crate) fn span_to_offsets(
    content: &str,
    start: &LineColumn,
    end: &LineColumn,
) -> (usize, usize) {
    let mut line_starts = Vec::new();
    let mut offset = 0usize;
    for line in content.split_inclusive('\n') {
        line_starts.push(offset);
        offset += line.len();
    }
    let start_line = (start.line.max(1) - 1) as usize;
    let end_line = (end.line.max(1) - 1) as usize;
    let start_offset = line_starts.get(start_line).cloned().unwrap_or(0)
        + start.column.saturating_sub(1) as usize;
    let end_offset = line_starts.get(end_line).cloned().unwrap_or(start_offset)
        + end.column.saturating_sub(1) as usize;
    (start_offset, end_offset)
}


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


pub(crate) fn span_to_offsets(
    content: &str,
    start: &LineColumn,
    end: &LineColumn,
) -> (usize, usize) {
    let mut line_starts = Vec::new();
    let mut offset = 0usize;
    for line in content.split_inclusive('\n') {
        line_starts.push(offset);
        offset += line.len();
    }
    let start_line = (start.line.max(1) - 1) as usize;
    let end_line = (end.line.max(1) - 1) as usize;
    let start_offset = line_starts.get(start_line).cloned().unwrap_or(0)
        + start.column.saturating_sub(1) as usize;
    let end_offset = line_starts.get(end_line).cloned().unwrap_or(start_offset)
        + end.column.saturating_sub(1) as usize;
    (start_offset, end_offset)
}


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


pub(crate) fn span_to_offsets(
    content: &str,
    start: &LineColumn,
    end: &LineColumn,
) -> (usize, usize) {
    let mut line_starts = Vec::new();
    let mut offset = 0usize;
    for line in content.split_inclusive('\n') {
        line_starts.push(offset);
        offset += line.len();
    }
    let start_line = (start.line.max(1) - 1) as usize;
    let end_line = (end.line.max(1) - 1) as usize;
    let start_offset = line_starts.get(start_line).cloned().unwrap_or(0)
        + start.column.saturating_sub(1) as usize;
    let end_offset = line_starts.get(end_line).cloned().unwrap_or(start_offset)
        + end.column.saturating_sub(1) as usize;
    (start_offset, end_offset)
}


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


pub(crate) fn span_to_offsets(
    content: &str,
    start: &LineColumn,
    end: &LineColumn,
) -> (usize, usize) {
    let mut line_starts = Vec::new();
    let mut offset = 0usize;
    for line in content.split_inclusive('\n') {
        line_starts.push(offset);
        offset += line.len();
    }
    let start_line = (start.line.max(1) - 1) as usize;
    let end_line = (end.line.max(1) - 1) as usize;
    let start_offset = line_starts.get(start_line).cloned().unwrap_or(0)
        + start.column.saturating_sub(1) as usize;
    let end_offset = line_starts.get(end_line).cloned().unwrap_or(start_offset)
        + end.column.saturating_sub(1) as usize;
    (start_offset, end_offset)
}
