# build.sh   (use this version – it uses tsconfig.json)
#!/usr/bin/env bash
set -euo pipefail

echo "Cleaning previous build..."
rm -rf dist src/gen
mkdir -p src/gen
mkdir -p dist/gen
mkdir -p dist/protobufjs
# mkdir -p src/protobufjs

# 1. Generate protobuf
CURRENT_DIR="$(pwd)"
PROTO_FILE="${CURRENT_DIR}/../schema/commands.proto"
OUTPUT_JS="${CURRENT_DIR}/src/gen/commands.js"

if [ ! -f "$PROTO_FILE" ]; then
    echo "ERROR: Proto file not found: $PROTO_FILE"
    exit 1
fi

echo "Generating protobuf JS..."
# Generate ESM static module with embedded minimal runtime and NO imports
npx pbjs -t static-module -w es6 -l minimal --no-imports "$PROTO_FILE" -o "$OUTPUT_JS"

echo "Generating TypeScript declarations..."
npx pbts "$OUTPUT_JS" -o src/gen/commands.d.ts

# 2. Build the ESM wrapper around the UMD minimal bundle
# The UMD sets globalThis.protobuf — we capture it and export as $protobuf
{
  echo "// Auto-generated ESM wrapper for protobufjs minimal (UMD -> ESM)"
  echo "const __scope = (function() {"
  echo "  var module = { exports: {} };"
  echo "  var exports = module.exports;"
  cat node_modules/protobufjs/dist/minimal/protobuf.js
  echo ""
  echo "  return module.exports;"
  echo "})();"
  echo "export default __scope;"
} > dist/protobufjs/minimal.esm.js

# 3. Compile with tsconfig
echo "Compiling TypeScript..."
npx tsc

if [ ! -f "dist/background.js" ] || [ ! -f "dist/content.js" ]; then
    echo "ERROR: .js files not generated"
    exit 1
fi

# 4. Copy generated files
cp src/gen/commands.js   dist/gen/commands.js
cp src/gen/commands.d.ts dist/gen/commands.d.ts 2>/dev/null || true

# 3b. Prepend $protobuf import into dist/gen/commands.js
# The file starts with /*minimal*/ then uses bare $protobuf — inject the import right after the comment
sed -i '/from "protobufjs\/minimal"/d' dist/gen/commands.js
sed -i '1a\import $protobuf from "../protobufjs/minimal.esm.js";' dist/gen/commands.js

cp src/sidepanel.html dist/ 2>/dev/null || true
cp dist/sidepanel.js  dist/ 2>/dev/null || true

# 5. Copy manifest
if [ ! -f "src/manifest.json" ]; then
    echo "ERROR: src/manifest.json not found"
    exit 1
fi
cp src/manifest.json dist/

# Content scripts and page scripts cannot be ES modules — strip any trailing export {}
sed -i '/^export {};$/d' dist/content.js
sed -i '/^export {};$/d' dist/page.js

echo ""
echo "Build finished."
echo "Load dist/ in chrome://extensions/"
echo ""

npx prettier --write "**/*.ts"
