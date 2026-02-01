"""
Demostración del pipeline de Fase 1.

Ejecutar con: python examples/demo_phase1.py
"""

from src.core.graph_utils import generate_sbm_graph
from src.core.curvature import compute_forman_ricci, get_curvature_distribution
from src.core.ricci_cleaning import ricci_clean, suggest_threshold
from src.core.routing import compare_routing, compute_graph_metrics


def main():
    print("=" * 50)
    print("FASE 1: Pipeline Baseline Ricci + Routing")
    print("=" * 50)
    
    # 1. Generar grafo SBM
    print("\n1. Generando grafo SBM (3 comunidades)...")
    G = generate_sbm_graph(sizes=[30, 30, 30], p_in=0.4, p_out=0.1, seed=42)
    print(f"   Nodos: {G.number_of_nodes()}")
    print(f"   Aristas: {G.number_of_edges()}")
    
    # 2. Calcular curvatura Forman-Ricci
    print("\n2. Calculando curvatura Forman-Ricci...")
    curvatures = compute_forman_ricci(G)
    dist = get_curvature_distribution(curvatures)
    print(f"   Min: {dist['min']:.2f}")
    print(f"   Max: {dist['max']:.2f}")
    print(f"   Mean: {dist['mean']:.2f}")
    print(f"   Std: {dist['std']:.2f}")
    
    # 3. Sugerir umbral
    print("\n3. Sugiriendo umbral de limpieza...")
    threshold = suggest_threshold(curvatures, method="percentile", percentile=15)
    print(f"   Umbral (percentil 15): {threshold:.2f}")
    
    # 4. Ricci-cleaning
    print("\n4. Aplicando Ricci-cleaning...")
    G_clean, info = ricci_clean(G, threshold=threshold, preserve_connectivity=True)
    print(f"   Aristas removidas: {info['edges_removed']}")
    print(f"   Aristas restantes: {G_clean.number_of_edges()}")
    print(f"   Puentes preservados: {info['bridges_preserved']}")
    print(f"   Componentes: {info['components_after']}")
    
    # 5. Comparar routing
    print("\n5. Comparando routing antes/después...")
    comparison = compare_routing(G, G_clean, n_queries=100, seed=42)
    print(f"   Routing antes: {comparison['before']['success_rate']:.1%}")
    print(f"   Routing después: {comparison['after']['success_rate']:.1%}")
    print(f"   Delta: {comparison['delta_success_rate']:+.1%}")
    print(f"   Stretch antes: {comparison['before']['avg_stretch']:.2f}")
    print(f"   Stretch después: {comparison['after']['avg_stretch']:.2f}")
    
    # 6. Métricas finales
    print("\n6. Métricas del grafo limpio:")
    metrics = compute_graph_metrics(G_clean)
    print(f"   Densidad: {metrics['density']:.3f}")
    print(f"   Clustering promedio: {metrics['avg_clustering']:.3f}")
    
    # Resumen
    print("\n" + "=" * 50)
    print("RESUMEN FASE 1")
    print("=" * 50)
    improvement = comparison['delta_success_rate'] > 0
    print(f"   ¿Mejora en routing? {'✓ SÍ' if improvement else '✗ NO'}")
    print(f"   ¿Grafo conexo? {'✓ SÍ' if info['components_after'] == 1 else '✗ NO'}")
    
    # Gate de aceptación
    gate_passed = info['components_after'] == 1
    print(f"\n   Gate Fase 1: {'✓ PASSED' if gate_passed else '✗ FAILED'}")


if __name__ == "__main__":
    main()
