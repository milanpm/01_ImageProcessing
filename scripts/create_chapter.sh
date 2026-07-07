#!/bin/bash

CHAPTER_NAME=$1

if [ -z "$CHAPTER_NAME" ]; then
  echo "Usage: ./scripts/create_chapter.sh 01_Image_Basics"
  exit 1
fi

CHAPTER_DIR="examples/$CHAPTER_NAME"

mkdir -p "$CHAPTER_DIR/docs" "$CHAPTER_DIR/images" "$CHAPTER_DIR/src"

cat > "$CHAPTER_DIR/README.md" <<EOF
# $CHAPTER_NAME

## Overview

This chapter explains practical image processing concepts and examples.

## Objectives

- Understand the core concept
- Implement examples using Python and OpenCV
- Apply the technique to machine vision problems

## Structure

\`\`\`text
$CHAPTER_NAME
├── README.md
├── docs/
├── images/
└── src/
\`\`\`

## Examples

To be added.

## Industrial Application

To be added.
EOF

cat > "$CHAPTER_DIR/docs/README.md" <<EOF
# Documents

Theory notes and technical explanations for $CHAPTER_NAME.
EOF

cat > "$CHAPTER_DIR/images/README.md" <<EOF
# Images

Sample images and result images for $CHAPTER_NAME.
EOF

cat > "$CHAPTER_DIR/src/README.md" <<EOF
# Source Code

Python/OpenCV source code for $CHAPTER_NAME.
EOF

echo "Created chapter: $CHAPTER_DIR"