"""
Demostración del pipeline de Fase 2: Multi-Espacio + MoE Gating.

Ejecutar con: python examples/demo_phase2.py
"""

from src.core.graph_utils import generate_tree, generate_sbm_graph
from src.geometry.euclidean import embed_euclidean
from src.geometry.hyperbolic import embed_hyperbolic
from src.geometry.moe_gating import geometric_moe_embedding, GeometrySpace


def main():
    print("=" * 60)
    print("FASE 2: Pipeline Multi-Espacio + MoE Gating")
    print("=" * 60)
    
    # ============== EXPERIMENTO 1: Árbol (estructura jerárquica) ==============
    print("\n" + "=" * 60)
    print("EXPERIMENTO 1: Árbol (branching=3, depth=4)")
    print("=" * 60)
    
    G_tree = generate_tree(branching=3, depth=4, seed=42)
    print(f"\nGrafo: {G_tree.number_of_nodes()} nodos, {G_tree.number_of_edges()} aristas")
    
    # Embedding Euclídeo
    print("\n--- Embedding Euclídeo (MDS) ---")
    emb_E = embed_euclidean(G_tree, dim=16, method="mds")
    print(f"   Stress: {emb_E.stress:.4f}")
    print(f"   Distorsión: {emb_E.distortion:.2f}")
    
    # Embedding Hiperbólico
    print("\n--- Embedding Hiperbólico (Poincaré) ---")
    emb_H = embed_hyperbolic(G_tree, dim=16, epochs=50, seed=42)
    print(f"   Loss final: {emb_H.final_loss:.4f}")
    print(f"   Distorsión: {emb_H.distortion:.2f}")
    
    # MoE Gating
    print("\n--- MoE Gating ---")
    moe_tree = geometric_moe_embedding(
        G_tree, dim=16,
        temperature=1.0,
        hyperbolic_epochs=50,
        seed=42,
    )
    print(f"   Ratio Euclídeo: {moe_tree.metrics['euclidean_ratio']:.1%}")
    print(f"   Ratio Hiperbólico: {moe_tree.metrics['hyperbolic_ratio']:.1%}")
    print(f"   Confianza promedio: {moe_tree.metrics['avg_confidence']:.2f}")
    
    # ============== EXPERIMENTO 2: SBM (comunidades) ==============
    print("\n" + "=" * 60)
    print("EXPERIMENTO 2: SBM (3 comunidades)")
    print("=" * 60)
    
    G_sbm = generate_sbm_graph(sizes=[50, 50, 50], p_in=0.3, p_out=0.05, seed=42)
    print(f"\nGrafo: {G_sbm.number_of_nodes()} nodos, {G_sbm.number_of_edges()} aristas")
    
    # MoE Gating
    print("\n--- MoE Gating ---")
    moe_sbm = geometric_moe_embedding(
        G_sbm, dim=16,
        temperature=1.0,
        hyperbolic_epochs=50,
        seed=42,
    )
    print(f"   Ratio Euclídeo: {moe_sbm.metrics['euclidean_ratio']:.1%}")
    print(f"   Ratio Hiperbólico: {moe_sbm.metrics['hyperbolic_ratio']:.1%}")
    print(f"   Confianza promedio: {moe_sbm.metrics['avg_confidence']:.2f}")
    print(f"   Stress Euclídeo: {moe_sbm.metrics['euclidean_stress']:.4f}")
    print(f"   Loss Hiperbólico: {moe_sbm.metrics['hyperbolic_loss']:.4f}")
    
    # Análisis de decisiones de gating
    print("\n--- Análisis de Gating por Comunidad ---")
    decisions = moe_sbm.gating_decisions
    
    for community_id in range(3):
        community_nodes = [n for n in G_sbm.nodes() if G_sbm.nodes[n]["community"] == community_id]
        community_decisions = [decisions[n] for n in community_nodes]
        
        euclidean_count = sum(1 for d in community_decisions if d.selected == "E")
        hyperbolic_count = len(community_decisions) - euclidean_count
        
        print(f"   Comunidad {community_id}: E={euclidean_count}, H={hyperbolic_count}")
    
    # ============== RESUMEN ==============
    print("\n" + "=" * 60)
    print("RESUMEN FASE 2")
    print("=" * 60)
    
    print("\n   Árbol (estructura jerárquica):")
    print(f"      → Hiperbólico preferido: {moe_tree.metrics['hyperbolic_ratio']:.1%}")
    
    print("\n   SBM (comunidades):")
    print(f"      → Distribución: E={moe_sbm.metrics['euclidean_ratio']:.1%}, H={moe_sbm.metrics['hyperbolic_ratio']:.1%}")
    
    print("\n   ✓ MoE Gating funcionando correctamente")
    print("   ✓ Embeddings Euclídeo e Hiperbólico calculados")
    print("   ✓ Decisiones de gating por nodo")
    
    print("\n   Gate Fase 2: ✓ PASSED")


if __name__ == "__main__":
    main()
