//! Benchmarks for Geometric Langlands implementation

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use geometric_langlands::prelude::*;

fn benchmark_reductive_group(c: &mut Criterion) {
    c.bench_function("reductive_group_gl_3", |b| {
        b.iter(|| {
            let _g = black_box(ReductiveGroup::gl_n(3));
        })
    });
}

fn benchmark_automorphic_form(c: &mut Criterion) {
    c.bench_function("eisenstein_series", |b| {
        b.iter(|| {
            let g = ReductiveGroup::gl_n(3);
            let _form = black_box(AutomorphicForm::eisenstein_series(&g, 2));
        })
    });
}

fn benchmark_hecke_operator(c: &mut Criterion) {
    c.bench_function("hecke_operator_apply", |b| {
        b.iter(|| {
            let g = ReductiveGroup::gl_n(3);
            let form = AutomorphicForm::eisenstein_series(&g, 2);
            let hecke = HeckeOperator::new(&g, 5);
            let _result = black_box(hecke.apply(&form));
        })
    });
}

criterion_group!(
    benches,
    benchmark_reductive_group,
    benchmark_automorphic_form,
    benchmark_hecke_operator
);
criterion_main!(benches);