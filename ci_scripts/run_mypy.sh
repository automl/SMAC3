MYPYPATH=smac/
MYPYOPTS="--ignore-missing-imports --strict"
MYPYOPTS="$MYPYOPS --disallow-any-unimported --disallow-any-expr --disallow-any-decorated --disallow-any-explicit --disallow-any-generics --disallow-untyped-defs"
mypy $MYPYOPTS $MYPYPATH
exit 0
