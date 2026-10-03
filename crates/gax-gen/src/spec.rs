//! Algebra declarations: the text format shared by the `algebra!` macro and the regeneration tool.
//!
//! ```text
//! algebra pga2d "Plane-based geometric algebra of the Euclidean plane";
//! basis e0 = 0, e1 = 1, e2 = 1;
//! kind Scalar = [1] "A scalar.";
//! kind Line = [e1, e2, e0] "A line `a x + b y + c = 0`.";
//! kind Point = [e20, e01, e12];
//! versor Motor = [1, e12, e20, e01];
//! alias Twist = Point;
//! ```
//!
//! * `basis` lists the basis vectors (named `e` plus one character) and their squares.
//! * `metric ea eb = v;` sets an off-diagonal metric entry (both `ea.eb` and `eb.ea`), which
//!   is how null bases such as CGA's `eo`, `ei` are declared.
//! * `kind` declares a typed subspace with its blade layout; `versor` does the same and also
//!   generates the fused sandwich kernels with it as the versor.
//! * `alias` declares a type alias for an existing kind.
//!
//! The format is designed to survive Rust tokenization, so the proc macro can pass its input
//! through `TokenStream::to_string()`.

use crate::algebra::{Algebra, AlgebraError};
use crate::table::Layout;

/// A declared kind.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KindSpec {
    /// Type name.
    pub name: String,
    /// Blade names as written.
    pub blades: Vec<String>,
    /// Parsed layout.
    pub layout: Layout,
    /// Whether sandwich kernels are generated with this kind as the versor.
    pub versor: bool,
    /// Whether operations may take this kind as their result type. False for a `part`: a
    /// kind reached only by name (casts, constructors, slots), so declaring it changes no
    /// existing result type (a PGA2D translator's log stays a `Point`, not a `Direction`).
    pub result: bool,
    /// Documentation.
    pub doc: String,
}

/// A parsed algebra declaration.
#[derive(Clone, Debug)]
pub struct AlgebraSpec {
    /// Module name.
    pub name: String,
    /// Documentation.
    pub doc: String,
    /// The algebra.
    pub algebra: Algebra,
    /// Declared kinds, in order.
    pub kinds: Vec<KindSpec>,
    /// Type aliases `(alias, kind)`.
    pub aliases: Vec<(String, String)>,
}

#[derive(Clone, Debug, PartialEq)]
enum Tok {
    Ident(String),
    Int(i64),
    Str(String),
    Punct(char),
}

fn tokenize(src: &str) -> Result<Vec<Tok>, AlgebraError> {
    let mut out = Vec::new();
    let chars: Vec<char> = src.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if c.is_whitespace() {
            i += 1;
        } else if c == '/' && chars.get(i + 1) == Some(&'/') {
            while i < chars.len() && chars[i] != '\n' {
                i += 1;
            }
        } else if c == '"' {
            let mut s = String::new();
            i += 1;
            loop {
                match chars.get(i) {
                    None => return Err(AlgebraError("unterminated string".into())),
                    Some('"') => {
                        i += 1;
                        break;
                    }
                    Some('\\') => {
                        match chars.get(i + 1) {
                            Some('n') => s.push('\n'),
                            Some(&e) => s.push(e),
                            None => return Err(AlgebraError("unterminated string".into())),
                        }
                        i += 2;
                    }
                    Some(&ch) => {
                        s.push(ch);
                        i += 1;
                    }
                }
            }
            out.push(Tok::Str(s));
        } else if c.is_ascii_digit() {
            let start = i;
            while i < chars.len() && chars[i].is_ascii_alphanumeric() {
                i += 1;
            }
            let text: String = chars[start..i].iter().collect();
            let v = text
                .parse::<i64>()
                .map_err(|_| AlgebraError(format!("bad number {text:?}")))?;
            out.push(Tok::Int(v));
        } else if c.is_ascii_alphabetic() || c == '_' {
            let start = i;
            while i < chars.len() && (chars[i].is_ascii_alphanumeric() || chars[i] == '_') {
                i += 1;
            }
            out.push(Tok::Ident(chars[start..i].iter().collect()));
        } else if "[]=;,-".contains(c) {
            out.push(Tok::Punct(c));
            i += 1;
        } else {
            return Err(AlgebraError(format!("unexpected character {c:?}")));
        }
    }
    Ok(out)
}

struct Parser {
    toks: Vec<Tok>,
    pos: usize,
}

impl Parser {
    fn peek(&self) -> Option<&Tok> {
        self.toks.get(self.pos)
    }
    fn next(&mut self) -> Option<Tok> {
        let t = self.toks.get(self.pos).cloned();
        self.pos += 1;
        t
    }
    fn expect_punct(&mut self, c: char) -> Result<(), AlgebraError> {
        match self.next() {
            Some(Tok::Punct(p)) if p == c => Ok(()),
            other => Err(AlgebraError(format!("expected `{c}`, found {other:?}"))),
        }
    }
    fn ident(&mut self) -> Result<String, AlgebraError> {
        match self.next() {
            Some(Tok::Ident(s)) => Ok(s),
            other => Err(AlgebraError(format!("expected a name, found {other:?}"))),
        }
    }
    fn int(&mut self) -> Result<i64, AlgebraError> {
        let neg = if self.peek() == Some(&Tok::Punct('-')) {
            self.pos += 1;
            true
        } else {
            false
        };
        match self.next() {
            Some(Tok::Int(v)) => Ok(if neg { -v } else { v }),
            other => Err(AlgebraError(format!(
                "expected an integer, found {other:?}"
            ))),
        }
    }
    fn opt_str(&mut self) -> String {
        if let Some(Tok::Str(s)) = self.peek() {
            let s = s.clone();
            self.pos += 1;
            s
        } else {
            String::new()
        }
    }
    fn blade(&mut self) -> Result<String, AlgebraError> {
        match self.next() {
            Some(Tok::Int(1)) => Ok("1".into()),
            Some(Tok::Ident(s)) => Ok(s),
            other => Err(AlgebraError(format!("expected a blade, found {other:?}"))),
        }
    }
}

fn is_type_name(s: &str) -> bool {
    s.chars().next().is_some_and(|c| c.is_ascii_uppercase())
}

impl AlgebraSpec {
    /// Parse a declaration.
    ///
    /// # Errors
    /// For any syntax error, unknown or duplicate names, reserved kind names, an invalid metric,
    /// or an invalid blade.
    #[allow(clippy::too_many_lines)]
    pub fn parse(src: &str) -> Result<AlgebraSpec, AlgebraError> {
        let mut p = Parser {
            toks: tokenize(src)?,
            pos: 0,
        };
        let mut name = None;
        let mut doc = String::new();
        let mut basis: Vec<(char, i64)> = Vec::new();
        let mut off_diagonal: Vec<(char, char, i64)> = Vec::new();
        let mut raw_kinds: Vec<(String, Vec<String>, bool, bool, String)> = Vec::new();
        let mut aliases = Vec::new();
        while let Some(tok) = p.next() {
            let Tok::Ident(word) = tok else {
                return Err(AlgebraError(format!(
                    "expected a declaration, found {tok:?}"
                )));
            };
            match word.as_str() {
                "algebra" => {
                    if name.is_some() {
                        return Err(AlgebraError("duplicate `algebra` declaration".into()));
                    }
                    name = Some(p.ident()?);
                    doc = p.opt_str();
                    p.expect_punct(';')?;
                }
                "basis" => {
                    if !basis.is_empty() {
                        return Err(AlgebraError("duplicate `basis` declaration".into()));
                    }
                    loop {
                        let v = p.ident()?;
                        let suffix = basis_suffix(&v)?;
                        p.expect_punct('=')?;
                        let sq = p.int()?;
                        basis.push((suffix, sq));
                        match p.next() {
                            Some(Tok::Punct(',')) => {}
                            Some(Tok::Punct(';')) => break,
                            other => {
                                return Err(AlgebraError(format!(
                                    "expected `,` or `;`, found {other:?}"
                                )));
                            }
                        }
                    }
                }
                "metric" => {
                    let a = basis_suffix(&p.ident()?)?;
                    let b = basis_suffix(&p.ident()?)?;
                    p.expect_punct('=')?;
                    let v = p.int()?;
                    p.expect_punct(';')?;
                    off_diagonal.push((a, b, v));
                }
                "kind" | "versor" | "part" => {
                    let kname = p.ident()?;
                    if !is_type_name(&kname) {
                        return Err(AlgebraError(format!(
                            "kind name {kname:?} must start with an uppercase letter"
                        )));
                    }
                    if crate::emit::RESERVED.contains(&kname.as_str()) {
                        return Err(AlgebraError(format!(
                            "kind name {kname:?} is reserved by gax"
                        )));
                    }
                    p.expect_punct('=')?;
                    p.expect_punct('[')?;
                    let mut blades = vec![p.blade()?];
                    loop {
                        match p.next() {
                            Some(Tok::Punct(',')) => blades.push(p.blade()?),
                            Some(Tok::Punct(']')) => break,
                            other => {
                                return Err(AlgebraError(format!(
                                    "expected `,` or `]`, found {other:?}"
                                )));
                            }
                        }
                    }
                    let kdoc = p.opt_str();
                    p.expect_punct(';')?;
                    raw_kinds.push((kname, blades, word == "versor", word != "part", kdoc));
                }
                "alias" => {
                    let a = p.ident()?;
                    p.expect_punct('=')?;
                    let k = p.ident()?;
                    p.expect_punct(';')?;
                    aliases.push((a, k));
                }
                other => return Err(AlgebraError(format!("unknown declaration `{other}`"))),
            }
        }
        let name = name.ok_or_else(|| AlgebraError("missing `algebra` declaration".into()))?;
        if basis.is_empty() {
            return Err(AlgebraError("missing `basis` declaration".into()));
        }
        let n = basis.len();
        let names: String = basis.iter().map(|b| b.0).collect();
        let mut metric = vec![vec![0i64; n]; n];
        for (i, &(_, sq)) in basis.iter().enumerate() {
            metric[i][i] = sq;
        }
        for (a, b, v) in off_diagonal {
            let ia = basis
                .iter()
                .position(|x| x.0 == a)
                .ok_or_else(|| AlgebraError(format!("metric: unknown e{a}")))?;
            let ib = basis
                .iter()
                .position(|x| x.0 == b)
                .ok_or_else(|| AlgebraError(format!("metric: unknown e{b}")))?;
            if ia == ib {
                return Err(AlgebraError(
                    "metric: use `basis` for diagonal entries".into(),
                ));
            }
            metric[ia][ib] = v;
            metric[ib][ia] = v;
        }
        let algebra = Algebra::new(&names, metric)?;
        let mut kinds: Vec<KindSpec> = Vec::new();
        for (kname, blades, versor, result, kdoc) in raw_kinds {
            if kinds.iter().any(|k| k.name == kname) {
                return Err(AlgebraError(format!("duplicate kind {kname}")));
            }
            let refs: Vec<&str> = blades.iter().map(String::as_str).collect();
            let layout = Layout::parse(&algebra, &refs)?;
            kinds.push(KindSpec {
                name: kname,
                blades,
                layout,
                versor,
                result,
                doc: kdoc,
            });
        }
        // Every algebra has a scalar kind and a kind holding every blade; add them (in the
        // canonical blade order: by grade, then by index) when the declaration does not.
        if !kinds.iter().any(|k| k.layout.blades == [(0, 1)]) {
            if kinds.iter().any(|k| k.name == "Scalar") {
                return Err(AlgebraError(
                    "a kind named `Scalar` must be the scalar kind `[1]`".into(),
                ));
            }
            kinds.insert(
                0,
                KindSpec {
                    name: "Scalar".into(),
                    blades: vec!["1".into()],
                    layout: Layout {
                        blades: vec![(0, 1)],
                    },
                    versor: false,
                    result: true,
                    doc: "A scalar.".into(),
                },
            );
        }
        let full = algebra.blade_count();
        if !kinds.iter().any(|k| k.layout.len() == full) {
            if kinds.iter().any(|k| k.name == "Multivector") {
                return Err(AlgebraError(
                    "a kind named `Multivector` must contain every blade".into(),
                ));
            }
            let mut masks: Vec<u32> = (0..full as u32).collect();
            masks.sort_by_key(|m| (m.count_ones(), *m));
            kinds.push(KindSpec {
                name: "Multivector".into(),
                blades: masks.iter().map(|&m| algebra.blade_name(m)).collect(),
                layout: Layout {
                    blades: masks.iter().map(|&m| (m, 1)).collect(),
                },
                versor: false,
                result: true,
                doc: "A general multivector, blades by grade then index.".into(),
            });
        }
        for (a, k) in &aliases {
            if !kinds.iter().any(|x| &x.name == k) {
                return Err(AlgebraError(format!(
                    "alias {a} refers to unknown kind {k}"
                )));
            }
            if kinds.iter().any(|x| &x.name == a) {
                return Err(AlgebraError(format!("alias {a} clashes with a kind")));
            }
        }
        Ok(AlgebraSpec {
            name,
            doc,
            algebra,
            kinds,
            aliases,
        })
    }

    /// The scalar kind.
    pub fn scalar_kind(&self) -> &KindSpec {
        self.kinds
            .iter()
            .find(|k| k.layout.blades == [(0, 1)])
            .expect("validated")
    }

    /// The declared kind for a result with blades `support`: among the kinds containing it,
    /// one that adds no grade the support lacks if possible (a bivector stays a bivector
    /// kind even when a smaller even kind would hold it), then the smallest, then the first
    /// declared.
    pub fn kind_for_support(&self, support: &std::collections::BTreeSet<u32>) -> Option<&KindSpec> {
        let grades: std::collections::BTreeSet<u32> =
            support.iter().map(|m| m.count_ones()).collect();
        self.kinds
            .iter()
            .filter(|k| k.result && support.is_subset(&k.layout.support()))
            .min_by_key(|k| (!k.layout.grades().is_subset(&grades), k.layout.len()))
    }
}

fn basis_suffix(name: &str) -> Result<char, AlgebraError> {
    let mut chars = name.chars();
    match (chars.next(), chars.next(), chars.next()) {
        (Some('e'), Some(c), None) => Ok(c),
        _ => Err(AlgebraError(format!(
            "basis vector {name:?} must be named `e` followed by one character"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A `part` is never chosen as a result type, even where it is the smallest kind that fits.
    #[test]
    fn parts_are_not_results() {
        let s = AlgebraSpec::parse(
            r#"algebra p2 "p"; basis e0 = 0, e1 = 1, e2 = 1;
            versor Point = [e20, e01, e12]; part Direction = [e20, e01];"#,
        )
        .expect("parse");
        let ideal: std::collections::BTreeSet<u32> = [0b101, 0b011].into_iter().collect();
        assert_eq!(s.kind_for_support(&ideal).expect("a kind").name, "Point");
        let d = s
            .kinds
            .iter()
            .find(|k| k.name == "Direction")
            .expect("declared");
        assert!(!d.result && !d.versor);
    }

    const PGA2D: &str = r#"
        algebra pga2d "PGA2D";
        basis e0 = 0, e1 = 1, e2 = 1;
        kind Scalar = [1];
        kind Line = [e1, e2, e0] "A line.";
        kind Point = [e20, e01, e12];
        versor Motor = [1, e12, e20, e01];
        kind Multivector = [1, e0, e1, e2, e01, e20, e12, e012];
        alias Twist = Point;
    "#;

    #[test]
    fn parses_pga2d() {
        let s = AlgebraSpec::parse(PGA2D).unwrap();
        assert_eq!(s.name, "pga2d");
        assert_eq!(s.kinds.len(), 5);
        assert!(s.kinds[3].versor);
        assert_eq!(s.kinds[2].layout.blades[0].1, -1); // e20 = -e02
        assert_eq!(s.aliases, vec![("Twist".to_string(), "Point".to_string())]);
    }

    #[test]
    fn parses_token_stream_spacing() {
        // What `TokenStream::to_string()` produces.
        let s = "algebra cga1 ; basis e1 = 1 , eo = 0 , ei = 0 ; metric eo ei = - 1 ; kind Scalar = [1] ; \
                 kind Multivector = [1 , e1 , eo , ei , e1o , e1i , eoi , e1oi] ;";
        let spec = AlgebraSpec::parse(s).unwrap();
        assert_eq!(spec.algebra.metric()[1][2], -1);
    }

    #[test]
    fn rejects_bad_input() {
        assert!(AlgebraSpec::parse("algebra x; basis e1 = 1; kind scalar = [1];").is_err());
        // A missing full kind is added.
        let auto =
            AlgebraSpec::parse("algebra x; basis e1 = 1, e2 = 1; kind Vector = [e1, e2];").unwrap();
        assert_eq!(auto.kinds.first().unwrap().name, "Scalar");
        assert_eq!(auto.kinds.last().unwrap().layout.len(), 4);
        assert!(AlgebraSpec::parse("algebra x; basis ex1 = 1;").is_err());
        assert!(
            AlgebraSpec::parse(
                "algebra x; basis e1 = 1; kind S = [1]; kind M = [1, e1]; alias S = M;"
            )
            .is_err()
        );
    }
}
