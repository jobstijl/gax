#!/usr/bin/env bash
# The game's dependency contract (see README.md): Bevy is plumbing only, and no other
# linear-algebra crate is used. Fails if a forbidden crate is in the dependency tree.
set -euo pipefail
cd "$(dirname "$0")/.."
forbidden=(
  bevy_audio bevy_render bevy_core_pipeline bevy_sprite bevy_sprite_render bevy_pbr
  bevy_ui bevy_ui_render bevy_ui_widgets bevy_text bevy_gizmos bevy_gizmos_render
  bevy_camera bevy_mesh bevy_image
  nalgebra cgmath ultraviolet vek euclid mint
)
# Exceptions: a forbidden crate allowed only below the given crate. `bevy_window` needs
# `bevy_image` for cursor icons, and `bevy_image` brings `euclid` through its texture packer.
declare -A only_under=(
  [bevy_image]="bevy_window bevy_winit bevy_internal"
  [euclid]="guillotiere"
)
tree=$(cargo tree --prefix none --edges normal,build --format '{p}' 2>/dev/null | awk '{print $1}' | sort -u)
bad=0
for c in "${forbidden[@]}"; do
  grep -qx "$c" <<<"$tree" || continue
  if [ -n "${only_under[$c]:-}" ]; then
    # The direct dependents of `c` must all be in the allowed list.
    parents=$(cargo tree -i "$c" --edges normal,build --prefix none --depth 1 --format '{p}' | awk 'NR > 1 {print $1}' | sort -u)
    ok=1
    for p in $parents; do
      grep -qw "$p" <<<"${only_under[$c]}" || ok=0
    done
    if [ "$ok" = 1 ]; then
      echo "allowed: $c (only under: $(echo $parents | tr '\n' ' '))"
      continue
    fi
  fi
  echo "forbidden crate in the dependency tree: $c"
  cargo tree -i "$c" --edges normal,build | head -20
  bad=1
done
[ "$bad" = 0 ] && echo "dependency contract ok"
exit $bad
