#!/bin/bash

# Create output directory
mkdir -p blog_posts/html

# Convert each Markdown file to HTML
for file in blog_posts/posts/*.md; do
    if [ -f "$file" ]; then
        filename=$(basename "$file" .md)
        echo "Converting $filename.md to HTML..."
        
        # Use pandoc to convert Markdown to HTML with better math support
        pandoc "$file" \
            --template=blog_posts/template.html \
            --output="blog_posts/html/${filename}.html" \
            --mathjax \
            --standalone \
            --from markdown+tex_math_dollars+tex_math_single_backslash
    fi
done

echo "Blog build complete! HTML files are in blog_posts/html/" 