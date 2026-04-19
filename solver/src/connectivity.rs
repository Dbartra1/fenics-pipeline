// src/connectivity.rs
//
// Element-to-node and element-to-DOF mappings for the structured hex grid.
//
// The 8 nodes of a hex element are ordered: bottom face CCW then top face CCW
// when viewed from outside (standard isoparametric hex convention).
// This ordering must be consistent with ke_base.rs — if you ever change it
// here you must recompute Ke.
//
//        7 ── 6
//       /|   /|   top face (iz+1)
//      4 ── 5 |
//      | 3 ─| 2
//      |/   |/    bottom face (iz)
//      0 ── 1
//
// Node local indices:
//   0:(0,0,0)  1:(1,0,0)  2:(1,1,0)  3:(0,1,0)   <- bottom CCW
//   4:(0,0,1)  5:(1,0,1)  6:(1,1,1)  7:(0,1,1)   <- top CCW

use crate::types::Grid;

// ─── Single-element functions (used in tests and ke_base) ────────────────────

/// Returns the 8 global node indices for the element at grid position (ix,iy,iz).
/// Order: bottom face CCW then top face CCW (see diagram above).
pub fn element_nodes(grid: &Grid, ix: usize, iy: usize, iz: usize) -> [usize; 8] {
    let n = |dx, dy, dz| grid.node_idx(ix + dx, iy + dy, iz + dz);
    [
        n(0, 0, 0), n(1, 0, 0), n(1, 1, 0), n(0, 1, 0), // bottom face
        n(0, 0, 1), n(1, 0, 1), n(1, 1, 1), n(0, 1, 1), // top face
    ]
}

/// Returns the 24 global DOF indices for a set of 8 element nodes.
/// DOFs are interleaved x/y/z per node: [3n, 3n+1, 3n+2] for each node n.
pub fn element_dofs(nodes: &[usize; 8]) -> [usize; 24] {
    let mut dofs = [0usize; 24];
    for (i, &n) in nodes.iter().enumerate() {
        dofs[3 * i]     = 3 * n;       // x
        dofs[3 * i + 1] = 3 * n + 1;   // y
        dofs[3 * i + 2] = 3 * n + 2;   // z
    }
    dofs
}

// ─── Precomputed maps for the full grid ──────────────────────────────────────

/// For each element, its 8 global node indices.
/// Index: connectivity[elem_idx] = [n0, n1, ..., n7]
pub fn precompute_connectivity(grid: &Grid) -> Vec<[usize; 8]> {
    let mut conn = vec![[0usize; 8]; grid.n_elem()];
    for iz in 0..grid.nz {
        for iy in 0..grid.ny {
            for ix in 0..grid.nx {
                let e = grid.elem_idx(ix, iy, iz);
                conn[e] = element_nodes(grid, ix, iy, iz);
            }
        }
    }
    conn
}

/// For each element, its 24 global DOF indices.
/// Index: dof_map[elem_idx] = [d0, d1, ..., d23]
pub fn precompute_dof_map(grid: &Grid) -> Vec<[usize; 24]> {
    let mut dof_map = vec![[0usize; 24]; grid.n_elem()];
    for iz in 0..grid.nz {
        for iy in 0..grid.ny {
            for ix in 0..grid.nx {
                let e = grid.elem_idx(ix, iy, iz);
                let nodes = element_nodes(grid, ix, iy, iz);
                dof_map[e] = element_dofs(&nodes);
            }
        }
    }
    dof_map
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Grid;

    // 2×1×1 grid: 2 elements side by side along x.
    // Hand-computed node layout:
    //
    //   Bottom face (iz=0):          Top face (iz=1):
    //   3 ── 4 ── 5                  9 ── 10 ── 11
    //   |  e0 | e1 |                 |  e0 |  e1 |
    //   0 ── 1 ── 2                  6 ──  7 ──  8
    //
    // node_idx(ix,iy,iz) = ix + iy*3 + iz*6   (nx=2, ny=1 → nx+1=3, (nx+1)(ny+1)=6)
    fn two_elem_grid() -> Grid {
        Grid { nx: 2, ny: 1, nz: 1, voxel_size: 0.001 }
    }

    // ── element_nodes: hand-verified against the diagram above ───────────────

    #[test]
    fn element_nodes_e0_bottom_left() {
        let g = two_elem_grid();
        let nodes = element_nodes(&g, 0, 0, 0);
        // Bottom CCW: (0,0,0)=0, (1,0,0)=1, (1,1,0)=4, (0,1,0)=3
        // Top    CCW: (0,0,1)=6, (1,0,1)=7, (1,1,1)=10,(0,1,1)=9
        assert_eq!(nodes, [0, 1, 4, 3, 6, 7, 10, 9]);
    }

    #[test]
    fn element_nodes_e1_right() {
        let g = two_elem_grid();
        let nodes = element_nodes(&g, 1, 0, 0);
        // Bottom CCW: (1,0,0)=1, (2,0,0)=2, (2,1,0)=5, (1,1,0)=4
        // Top    CCW: (1,0,1)=7, (2,0,1)=8, (2,1,1)=11,(1,1,1)=10
        assert_eq!(nodes, [1, 2, 5, 4, 7, 8, 11, 10]);
    }

    #[test]
    fn adjacent_elements_share_exactly_four_nodes() {
        // e0 and e1 share the face at ix=1: nodes {1, 4, 7, 10}
        let g = two_elem_grid();
        let n0: std::collections::HashSet<usize> = element_nodes(&g, 0, 0, 0).into();
        let n1: std::collections::HashSet<usize> = element_nodes(&g, 1, 0, 0).into();
        let shared: Vec<_> = n0.intersection(&n1).collect();
        assert_eq!(shared.len(), 4, "adjacent elements must share exactly 4 nodes");
    }

    #[test]
    fn corner_elements_share_exactly_one_node() {
        // Elements diagonal in all three axes share exactly 1 node (a corner).
        // Requires 3D diagonal — a 2×2×1 grid gives a shared edge in Z, not a corner.
        let g = Grid { nx: 2, ny: 2, nz: 2, voxel_size: 0.001 };
        let n00: std::collections::HashSet<usize> = element_nodes(&g, 0, 0, 0).into();
        let n11: std::collections::HashSet<usize> = element_nodes(&g, 1, 1, 1).into();
        let shared: Vec<_> = n00.intersection(&n11).collect();
        assert_eq!(shared.len(), 1, "3D-diagonal elements share exactly 1 node");
    }

    // ── element_dofs: structural checks ──────────────────────────────────────

    #[test]
    fn element_dofs_length_is_24() {
        let g = two_elem_grid();
        let nodes = element_nodes(&g, 0, 0, 0);
        let dofs = element_dofs(&nodes);
        assert_eq!(dofs.len(), 24);
    }

    #[test]
    fn element_dofs_interleaving_correct() {
        // For node with global index N, DOFs must be [3N, 3N+1, 3N+2].
        let g = two_elem_grid();
        let nodes = element_nodes(&g, 0, 0, 0);
        let dofs = element_dofs(&nodes);
        for (i, &n) in nodes.iter().enumerate() {
            assert_eq!(dofs[3 * i],     3 * n,     "node {i} x-DOF wrong");
            assert_eq!(dofs[3 * i + 1], 3 * n + 1, "node {i} y-DOF wrong");
            assert_eq!(dofs[3 * i + 2], 3 * n + 2, "node {i} z-DOF wrong");
        }
    }

    #[test]
    fn adjacent_elements_share_exactly_twelve_dofs() {
        // 4 shared nodes × 3 DOFs each = 12.
        let g = two_elem_grid();
        let d0: std::collections::HashSet<usize> =
            element_dofs(&element_nodes(&g, 0, 0, 0)).into();
        let d1: std::collections::HashSet<usize> =
            element_dofs(&element_nodes(&g, 1, 0, 0)).into();
        let shared: Vec<_> = d0.intersection(&d1).collect();
        assert_eq!(shared.len(), 12);
    }

    // ── precompute_connectivity: full-grid properties ─────────────────────────

    #[test]
    fn connectivity_length_equals_n_elem() {
        let g = Grid { nx: 4, ny: 3, nz: 2, voxel_size: 0.001 };
        let conn = precompute_connectivity(&g);
        assert_eq!(conn.len(), g.n_elem());
    }

    #[test]
    fn connectivity_all_nodes_in_valid_range() {
        let g = Grid { nx: 4, ny: 3, nz: 2, voxel_size: 0.001 };
        let conn = precompute_connectivity(&g);
        for (e, nodes) in conn.iter().enumerate() {
            for &n in nodes {
                assert!(n < g.n_nodes(), "elem {e}: node {n} >= n_nodes {}", g.n_nodes());
            }
        }
    }

    #[test]
    fn connectivity_every_node_appears_at_least_once() {
        // On a complete grid every node belongs to at least one element.
        let g = Grid { nx: 3, ny: 2, nz: 2, voxel_size: 0.001 };
        let conn = precompute_connectivity(&g);
        let mut seen = vec![false; g.n_nodes()];
        for nodes in &conn {
            for &n in nodes {
                seen[n] = true;
            }
        }
        let unvisited: Vec<usize> = seen.iter().enumerate()
            .filter(|(_, &v)| !v)
            .map(|(i, _)| i)
            .collect();
        assert!(unvisited.is_empty(), "nodes not in any element: {unvisited:?}");
    }

    #[test]
    fn connectivity_matches_direct_element_nodes() {
        // precompute_connectivity must agree with calling element_nodes directly.
        let g = Grid { nx: 4, ny: 3, nz: 2, voxel_size: 0.001 };
        let conn = precompute_connectivity(&g);
        for iz in 0..g.nz {
            for iy in 0..g.ny {
                for ix in 0..g.nx {
                    let e = g.elem_idx(ix, iy, iz);
                    assert_eq!(conn[e], element_nodes(&g, ix, iy, iz));
                }
            }
        }
    }

    // ── precompute_dof_map: structural checks ─────────────────────────────────

    #[test]
    fn dof_map_length_equals_n_elem() {
        let g = Grid { nx: 4, ny: 3, nz: 2, voxel_size: 0.001 };
        let dof_map = precompute_dof_map(&g);
        assert_eq!(dof_map.len(), g.n_elem());
    }

    #[test]
    fn dof_map_all_dofs_in_valid_range() {
        let g = Grid { nx: 4, ny: 3, nz: 2, voxel_size: 0.001 };
        let dof_map = precompute_dof_map(&g);
        for (e, dofs) in dof_map.iter().enumerate() {
            for &d in dofs {
                assert!(d < g.n_dof(), "elem {e}: dof {d} >= n_dof {}", g.n_dof());
            }
        }
    }

    #[test]
    fn dof_map_consistent_with_connectivity() {
        let g = Grid { nx: 4, ny: 3, nz: 2, voxel_size: 0.001 };
        let conn    = precompute_connectivity(&g);
        let dof_map = precompute_dof_map(&g);
        for e in 0..g.n_elem() {
            let expected = element_dofs(&conn[e]);
            assert_eq!(dof_map[e], expected, "dof_map[{e}] inconsistent with connectivity");
        }
    }
}