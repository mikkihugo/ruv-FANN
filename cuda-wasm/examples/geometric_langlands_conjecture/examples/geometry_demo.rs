// Demonstration of geometric structures for Langlands program
// Shows sheaves, bundles, moduli spaces, and invariants

use geometric_langlands::geometry::*;
use std::collections::HashMap;

fn main() {
    println!("🔷 Geometric Langlands Conjecture - Geometry Demo");
    println!("=" .repeat(60));
    
    // Part 1: Sheaf Theory
    println!("\n📚 Part 1: Sheaf Theory");
    demo_sheaves();
    
    // Part 2: G-Bundles
    println!("\n🎯 Part 2: G-Bundles and Connections");
    demo_bundles();
    
    // Part 3: Moduli Spaces
    println!("\n🌌 Part 3: Moduli Spaces");
    demo_moduli_spaces();
    
    // Part 4: Cohomology
    println!("\n🧮 Part 4: Cohomology Computation");
    demo_cohomology();
    
    // Part 5: Geometric Invariants
    println!("\n⚡ Part 5: Geometric Invariants");
    demo_invariants();
    
    println!("\n✅ Geometry demonstration completed!");
}

fn demo_sheaves() {
    use sheaf::*;
    
    // Create a perverse sheaf on a 4-dimensional variety
    let mut perverse = PerverseSheaf::<f64>::middle_perversity(4);
    
    // Add stratification
    let stratum1 = OpenSet {
        id: "open_stratum".to_string(),
        dimension: 4,
        is_affine: true,
    };
    let stratum2 = OpenSet {
        id: "singular_locus".to_string(),
        dimension: 2,
        is_affine: false,
    };
    
    perverse.add_stratum(stratum1.clone());
    perverse.add_stratum(stratum2);
    
    // Check perversity conditions
    let satisfies_perversity = perverse.check_perversity(0);
    
    println!("  • Created perverse sheaf on 4D variety");
    println!("  • Stratification has {} strata", perverse.stratification.len());
    println!("  • Perversity condition satisfied: {}", satisfies_perversity);
    
    // Add local sections
    let section = LocalSection {
        open_set: stratum1,
        data: 3.14159,
        restriction_maps: HashMap::new(),
    };
    
    perverse.sheaf.add_section(section);
    println!("  • Added local section with value π");
    println!("  • Global sections: {}", perverse.sheaf.global_sections().len());
}

fn demo_bundles() {
    use bundle::*;
    
    // Create a principal G-bundle with SL(2) structure group
    let mut bundle = GBundle::new(
        2, // Curve genus g=2
        StructureGroup::SL(2),
        BundleType::Vector(2)
    );
    
    // Add a connection
    let mut connection = ConnectionForm::new(2, StructureGroup::SL(2));
    connection.local_forms.insert(
        "patch_1".to_string(),
        vec![vec![0.0, 1.0], vec![-1.0, 0.0]]
    );
    connection.compute_curvature();
    
    bundle.set_connection(connection);
    
    // Check properties
    let is_stable = bundle.is_stable();
    let degree = bundle.degree();
    let rank = bundle.rank();
    let chern_classes = bundle.chern_classes();
    
    println!("  • Created SL(2)-bundle on genus 2 curve");
    println!("  • Rank: {}, Degree: {}", rank, degree);
    println!("  • Is stable: {}", is_stable);
    println!("  • First Chern class: {:.6}", chern_classes.get(1).unwrap_or(&0.0));
    
    // Create Higgs bundle
    let higgs_field = vec![
        vec![0.0, 1.0],
        vec![0.0, 0.0]
    ];
    let mut higgs = HiggsBundle::new(bundle, higgs_field);
    higgs.compute_spectral_curve();
    
    if let Some(ref curve) = higgs.spectral_curve {
        println!("  • Higgs bundle spectral curve: genus {}, degree {}", 
                curve.genus, curve.degree);
        println!("  • Ramification points: {}", curve.ramification_points.len());
    }
    
    println!("  • Higgs bundle is stable: {}", higgs.is_stable());
}

fn demo_moduli_spaces() {
    use moduli_space::*;
    
    // Create moduli space of stable bundles
    let mut moduli = ModuliSpace::stable_bundles(2, StructureGroup::SL(2));
    
    println!("  • Moduli space dimension: {}", moduli.dimension());
    
    // Add some points
    for i in 0..5 {
        let coords = vec![i as f64, (i * i) as f64 % 7.0, (i as f64).sin()];
        let point = ModuliPoint::new(coords);
        moduli.add_point(point);
    }
    
    println!("  • Added {} points to moduli space", moduli.points.len());
    
    // Count stable points
    let stable_count = moduli.points.iter()
        .filter(|p| p.stability_params.is_stable)
        .count();
    
    println!("  • Stable points: {}/{}", stable_count, moduli.points.len());
    
    // Compute Betti numbers
    let betti = moduli.betti_numbers();
    println!("  • Betti numbers: {:?}", &betti[..5.min(betti.len())]);
    
    // Create character variety (local systems)
    let local_sys = LocalSystemModuli::new(2, StructureGroup::GL(2));
    println!("  • Character variety dimension: {}", local_sys.character_variety_dim);
    
    // Simpson correspondence
    if let Some(point) = moduli.points.first() {
        if let Some(rep) = local_sys.simpson_correspondence(point) {
            println!("  • Simpson correspondence: Higgs ↔ Local system");
            println!("    - Irreducible: {}", rep.is_irreducible);
            println!("    - Unitary: {}", rep.is_unitary);
        }
    }
}

fn demo_cohomology() {
    use cohomology::*;
    
    // Create Čech complex
    let open_sets = vec![
        OpenSet { id: "U1".to_string(), dimension: 2, is_affine: true },
        OpenSet { id: "U2".to_string(), dimension: 2, is_affine: true },
        OpenSet { id: "U3".to_string(), dimension: 2, is_affine: true },
    ];
    
    let sheaf: Sheaf<f64> = Sheaf::new(2, sheaf::SheafType::Coherent);
    let cech = CechComplex::new(open_sets, sheaf);
    
    // Compute cohomology
    let h0 = cech.cohomology(0);
    let h1 = cech.cohomology(1);
    let h2 = cech.cohomology(2);
    
    println!("  • Čech cohomology dimensions:");
    println!("    - H⁰: {}", h0.dimension);
    println!("    - H¹: {}", h1.dimension);
    println!("    - H²: {}", h2.dimension);
    
    // Spectral sequence computation
    let mut spectral = SpectralSequence::leray_sequence(2, 2);
    println!("  • Leray spectral sequence (fiber bundle):");
    println!("    - Starting at E₂ page");
    
    while !spectral.has_converged() && spectral.page < 10 {
        spectral.next_page();
    }
    
    let total_h2 = spectral.total_cohomology(2);
    println!("    - Converged at page {}", spectral.page);
    println!("    - Total H²: dimension {}", total_h2.dimension);
    
    // Cup product
    let mut h1_again = CohomologyGroup::new(1);
    h1_again.add_generator(vec![1.0, 0.0]);
    
    let h2_cup = h1_again.cup_product(&h1_again);
    println!("  • Cup product H¹ ∪ H¹ → H²: dimension {}", h2_cup.dimension);
}

fn demo_invariants() {
    use invariants::*;
    
    // Create bundle for invariant computation
    let mut bundle = GBundle::new(
        2,
        StructureGroup::U(2),
        BundleType::Vector(2)
    );
    
    // Add connection with specific curvature
    let mut connection = ConnectionForm::new(2, StructureGroup::U(2));
    connection.curvature = Some(vec![
        vec![1.0, 0.5],
        vec![-0.5, 1.0]
    ]);
    bundle.set_connection(connection);
    
    // Compute all invariants
    let all_invariants = compute_all_invariants(&bundle);
    
    println!("  • Geometric invariants computed:");
    for (name, value) in &all_invariants {
        println!("    - {}: {:.6}", name, value);
    }
    
    // Chern classes
    let chern = ChernClass::from_bundle(&bundle);
    println!("  • Chern classes: {:?}", chern.classes.iter().map(|x| format!("{:.4}", x)).collect::<Vec<_>>());
    
    // Todd class and genus
    let todd = ToddClass::from_chern(&chern);
    println!("  • Todd genus: {:.6}", todd.genus);
    
    // Riemann-Roch computation
    let chi = todd.riemann_roch(&chern, 2);
    println!("  • Euler characteristic (Riemann-Roch): {}", chi);
    
    // K-theory class
    let k_class = KTheoryClass::from_bundle(&bundle);
    println!("  • K-theory rank: {}", k_class.rank);
    println!("  • Adams operations: {:?}", 
            k_class.adams_operations.iter().map(|x| format!("{:.4}", x)).collect::<Vec<_>>());
    
    // Index computation
    let curvature = bundle.connection.as_ref().unwrap().curvature.as_ref().unwrap();
    let index = IndexTheorem::atiyah_singer_index(curvature, &todd, 2);
    println!("  • Atiyah-Singer index: {}", index);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_demo_runs() {
        // Just test that the demo functions don't panic
        demo_sheaves();
        demo_bundles();
        demo_moduli_spaces();
        demo_cohomology();
        demo_invariants();
    }
}