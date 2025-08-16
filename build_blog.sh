#!/bin/bash

# Convert each Markdown file to HTML
for file in blog/markdown/*.md; do
    if [ -f "$file" ]; then
        filename=$(basename "$file" .md)
        echo "Converting $filename.md to HTML..."
        
        # Use pandoc to convert Markdown to HTML with better math support
        pandoc "$file" \
            --template=blog/template.html \
            --output="blog/posts/${filename}.html" \
            --mathjax \
            --standalone \
            --from markdown+tex_math_dollars+tex_math_single_backslash
    fi
done

echo "Blog build complete! HTML files are in blog/posts" 