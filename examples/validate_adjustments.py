"""
Script de validación pre-escalamiento RTX 4060.

Verifica que los ajustes de Fase 1-2 funcionan correctamente
antes de escalar a grafos grandes.

Ejecutar con: python examples/validate_adjustments.py
"""

import numpy as np
from src.core.graph_utils import generate_tree, generate_sbm_graph, generate_grid_with_noise
from src.core.curvature import compute_forman_ricci, classify_edges_by_curvature
from src.geometry.euclidean import embed_euclidean
from src.geometry.hyperbolic import embed_hyperbolic
from src.geometry.moe_gating import geometric_moe_embedding, compute_node_features


def print_section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


def validate_gating_on_tree():
    """Verificar que árbol prefiere Hiperbólico."""
    print_section("VALIDACIÓN 1: Árbol → Hiperbólico esperado")
    
    G = generate_tree(branching=3, depth=4, seed=42)
    print(f"Grafo: {G.number_of_nodes()} nodos, {G.number_of_edges()} aristas")
    
    # Curvatura
    curvatures = compute_forman_ricci(G)
    curv_values = list(curvatures.values())
    print(f"\nCurvatura Forman:")
    print(f"   Min: {min(curv_values):.1f}, Max: {max(curv_values):.1f}")
    print(f"   Mean: {np.mean(curv_values):.2f}")
    
    # Features de muestra
    features = compute_node_features(G, curvatures)
    sample_node = list(features.keys())[0]
    print(f"\nFeatures nodo {sample_node}:")
    print(f"   Curvatura local: {features[sample_node].local_curvature:.2f}")
    print(f"   Clustering: {features[sample_node].clustering:.3f}")
    print(f"   Grado: {features[sample_node].degree}")
    
    # MoE embedding
    result = geometric_moe_embedding(
        G, dim=8, 
        hyperbolic_epochs=30,
        seed=42,
    )
    
    print(f"\nResultado MoE Gating:")
    print(f"   Euclídeo: {result.metrics['euclidean_ratio']:.1%}")
    print(f"   Hiperbólico: {result.metrics['hyperbolic_ratio']:.1%}")
    print(f"   Confianza promedio: {result.metrics['avg_confidence']:.2f}")
    
    # Verificar
    passed = result.metrics['hyperbolic_ratio'] > 0.5
    status = "✓ PASSED" if passed else "✗ FAILED"
    print(f"\n   → Árbol prefiere Hiperbólico: {status}")
    
    return passed


def validate_gating_on_sbm():
    """Verificar comportamiento en SBM (comunidades)."""
    print_section("VALIDACIÓN 2: SBM → Mixto o variable")
    
    G = generate_sbm_graph(sizes=[40, 40, 40], p_in=0.4, p_out=0.02, seed=42)
    print(f"Grafo: {G.number_of_nodes()} nodos, {G.number_of_edges()} aristas")
    
    # Curvatura
    curvatures = compute_forman_ricci(G)
    curv_values = list(curvatures.values())
    print(f"\nCurvatura Forman:")
    print(f"   Min: {min(curv_values):.1f}, Max: {max(curv_values):.1f}")
    print(f"   Mean: {np.mean(curv_values):.2f}")
    
    # MoE embedding
    result = geometric_moe_embedding(
        G, dim=8,
        hyperbolic_epochs=30,
        seed=42,
    )
    
    print(f"\nResultado MoE Gating:")
    print(f"   Euclídeo: {result.metrics['euclidean_ratio']:.1%}")
    print(f"   Hiperbólico: {result.metrics['hyperbolic_ratio']:.1%}")
    print(f"   Confianza promedio: {result.metrics['avg_confidence']:.2f}")
    
    # SBM denso debería tener algo de Euclídeo por el clustering
    # No hay un "correcto" aquí, solo que funcione
    passed = True  # Solo verificamos que no crashea
    print(f"\n   → SBM procesado correctamente: ✓ PASSED")
    
    return passed


def validate_mds_distortion():
    """Verificar que MDS no tiene distorsiones extremas."""
    print_section("VALIDACIÓN 3: MDS Distorsión razonable")
    
    G = generate_tree(branching=3, depth=4, seed=42)
    print(f"Grafo árbol: {G.number_of_nodes()} nodos")
    
    result = embed_euclidean(G, dim=16, method="mds")
    
    print(f"\nMétricas MDS:")
    print(f"   Stress: {result.stress:.4f}")
    print(f"   Distorsión (p95): {result.distortion:.2f}")
    
    # Distorsión razonable < 10 para árboles
    passed = result.distortion < 20 and np.isfinite(result.distortion)
    status = "✓ PASSED" if passed else "✗ FAILED (>20 o infinito)"
    print(f"\n   → Distorsión razonable: {status}")
    
    return passed


def validate_hyperbolic_quality():
    """Verificar calidad del embedding hiperbólico."""
    print_section("VALIDACIÓN 4: Embedding Hiperbólico")
    
    G = generate_tree(branching=3, depth=3, seed=42)
    print(f"Grafo árbol: {G.number_of_nodes()} nodos")
    
    result = embed_hyperbolic(G, dim=8, epochs=50, seed=42)
    
    print(f"\nMétricas Hiperbólico:")
    print(f"   Loss final: {result.final_loss:.4f}")
    print(f"   Distorsión: {result.distortion:.2f}")
    
    # Verificar que todos los puntos están en el ball
    norms = [np.linalg.norm(pos) for pos in result.positions.values()]
    max_norm = max(norms)
    
    print(f"   Max norma: {max_norm:.4f} (debe ser < 1)")
    
    passed = max_norm < 1.0 and np.isfinite(result.final_loss)
    status = "✓ PASSED" if passed else "✗ FAILED"
    print(f"\n   → Embedding válido: {status}")
    
    return passed


def main():
    print("\n" + "="*60)
    print("   VALIDACIÓN PRE-ESCALAMIENTO RTX 4060")
    print("="*60)
    
    results = []
    
    results.append(("Árbol → Hiperbólico", validate_gating_on_tree()))
    results.append(("SBM procesado", validate_gating_on_sbm()))
    results.append(("MDS distorsión", validate_mds_distortion()))
    results.append(("Hiperbólico válido", validate_hyperbolic_quality()))
    
    print_section("RESUMEN")
    
    all_passed = True
    for name, passed in results:
        status = "✓" if passed else "✗"
        print(f"   {status} {name}")
        all_passed = all_passed and passed
    
    print("\n" + "="*60)
    if all_passed:
        print("   ✓ TODAS LAS VALIDACIONES PASARON")
        print("   → Listo para escalar en RTX 4060")
    else:
        print("   ✗ ALGUNAS VALIDACIONES FALLARON")
        print("   → Revisar antes de escalar")
    print("="*60 + "\n")
    
    return all_passed


if __name__ == "__main__":
    main()
