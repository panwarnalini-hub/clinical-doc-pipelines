# Databricks notebook source
# MAGIC %sql
# MAGIC USE CATALOG dev_clinical;
# MAGIC USE SCHEMA doc_test;

# COMMAND ----------

# STRUCTURE TO SILVER SECTIONS PIPELINE
#
# WHAT THIS NOTEBOOK DOES:
#   It takes the raw extraction from Bronze and turn it into real structure:
#   sections with hierarchy, clean boundaries, and meaning.
#
# HOW IT WORKS:
#   1) Read Bronze sections and pages
#   2) Build proper section hierarchy using TOC and headings
#   3) Remove repeated header/footer noise
#   4) Split long content into semantic chunks (helps recall later)
#   5) Create embeddings and classify sections more accurately
#   6) Write the upgraded structure into Silver tables
#
# WHY THIS MATTERS:
#   Structure will give context and downstream features depend on knowing "what belongs where"
#   so search results, analytics, and automation don't fall apart.
#
# OUTPUT:
#   Silver sections that are clean, organized, and ready
#   for everything intelligent that comes next.
#
# NOTE:
#   Bronze = full extraction of the document, exactly as it was written
#   Silver = cleaned and organized structure that software can understand
#
# PROCESSING ORDER:
#   Step 1: Load Bronze (sections + pages)
#   Step 2: Remove Header/Footer Noise
#   Step 3: Hierarchy Reconstruction (headings → parents)
#   Step 4: Merge Broken Paragraphs
#   Step 5: Semantic Chunking (AFTER hierarchy + merge)
#   Step 6: Add Structural Columns
#   Step 7: Write Silver Table

# COMMAND ----------

# MAGIC %md
# MAGIC # Configuration

# COMMAND ----------

from dataclasses import dataclass

# Configuration - mirrors 01_Document_Extraction_to_Bronze.py
@dataclass
class PipelineConfig:
    
    # Unity Catalog
    catalog: str = "dev_clinical"
    schema: str = "doc_test"
    
    # Bronze Tables (input)
    bronze_pages: str = "bronze_pages"
    bronze_sections: str = "bronze_sections"
    
    # Silver Tables (output)
    silver_sections: str = "silver_sections"
    
    # Processing thresholds
    header_footer_threshold: float = 0.5  # Text appearing on >50% of pages is noise
    paragraph_merge_max_words: int = 200  # Max words for merge candidates
    chunk_word_threshold: int = 200       # Split sections longer than this
    chunk_target_words: int = 150         # Target chunk size
    
    def full_table(self, table: str) -> str:
        return f"{self.catalog}.{self.schema}.{table}"


config = PipelineConfig()
print(f"Pipeline: {config.catalog}.{config.schema}")
print(f"Input:  {config.full_table(config.bronze_sections)}")
print(f"Output: {config.full_table(config.silver_sections)}")

# COMMAND ----------

# MAGIC %md
# MAGIC # Step 1: Load Bronze Data

# COMMAND ----------

from pyspark.sql import Window
from pyspark.sql import functions as F
from pyspark.sql.types import BooleanType, IntegerType, StringType

bronze_sections_df = spark.table(config.full_table(config.bronze_sections))
bronze_pages_df = spark.table(config.full_table(config.bronze_pages))

# Fix missing document_id
if "document_id" not in bronze_sections_df.columns and "protocol_id" in bronze_sections_df.columns:
    bronze_sections_df = bronze_sections_df.withColumn("document_id", F.col("protocol_id"))


# Add missing expected columns if they are not present
expected_columns = {
    "is_toc": F.lit(False).cast(BooleanType()),
    "heading_level": F.lit(None).cast(IntegerType()),
    "content_markdown": F.col("content_text"),   # fall back to content_text
    "is_strikethrough": F.lit(False).cast(BooleanType()),
}

for col, default_expr in expected_columns.items():
    if col not in bronze_sections_df.columns:
        bronze_sections_df = bronze_sections_df.withColumn(col, default_expr)

# Ensure section_type exists
if "section_type" not in bronze_sections_df.columns:
    bronze_sections_df = bronze_sections_df.withColumn("section_type",
        F.when(F.col("heading_level").isNotNull(), "heading").otherwise("paragraph")
    )

print("Bronze sections loaded:", bronze_sections_df.count())
print("Bronze pages loaded:", bronze_pages_df.count())

bronze_sections_df.cache()
bronze_pages_df.cache()


# COMMAND ----------

# MAGIC %md
# MAGIC # Step 2: Remove Header/Footer Noise
# MAGIC
# MAGIC Identify repeated strings appearing on many pages (> 50% of pages per document).
# MAGIC These are typically headers, footers, or watermarks that slipped through extraction.

# COMMAND ----------

# Get total page counts per document
page_counts_df = bronze_pages_df.groupBy("document_id").agg(
    F.countDistinct("page_number").alias("total_pages")
)

# Find content that repeats across many pages (noise candidates)
# We look for exact text matches appearing on >50% of pages
content_frequency_df = bronze_sections_df.groupBy(
    "document_id", 
    "content_text"
).agg(
    F.countDistinct("page_number").alias("page_appearances")
)

# Join to get threshold
noise_candidates_df = content_frequency_df.join(
    page_counts_df, 
    on="document_id"
).withColumn(
    "appearance_ratio",
    F.col("page_appearances") / F.col("total_pages")
).filter(
    F.col("appearance_ratio") > config.header_footer_threshold
).select(
    "document_id",
    "content_text"
)

# Also use explicit header/footer text from bronze_pages
explicit_noise_df = bronze_pages_df.select(
    "document_id",
    F.col("header_text").alias("content_text")
).filter(
    F.col("content_text").isNotNull() & (F.trim(F.col("content_text")) != "")
).union(
    bronze_pages_df.select(
        "document_id",
        F.col("footer_text").alias("content_text")
    ).filter(
        F.col("content_text").isNotNull() & (F.trim(F.col("content_text")) != "")
    )
).distinct()

# Combine noise sources
all_noise_df = noise_candidates_df.union(explicit_noise_df).distinct()

print(f"Noise patterns identified: {all_noise_df.count()}")

# Filter out noise from sections using LEFT ANTI JOIN
clean_sections_df = bronze_sections_df.join(
    all_noise_df,
    on=["document_id", "content_text"],
    how="left_anti"
)

print(f"Sections after noise removal: {clean_sections_df.count()} rows")
print(f"Removed: {bronze_sections_df.count() - clean_sections_df.count()} noise sections")

# COMMAND ----------

# MAGIC %md
# MAGIC # Step 3: Hierarchy Reconstruction (Headings to Parents)
# MAGIC
# MAGIC Build proper parent-child relationships using heading levels.
# MAGIC - Headings become parents
# MAGIC - Heading level determines depth (level 1 → top, level 2 → child of last level 1, etc.)
# MAGIC - Paragraphs before any heading → attach to synthetic root

# COMMAND ----------

# First, establish document order
ordered_sections_df = clean_sections_df.withColumn(
    "doc_order",
    F.row_number().over(
        Window.partitionBy("document_id")
        .orderBy("page_number", "sequence_number")
    )
)

# Register as temp view for hierarchy building
ordered_sections_df.createOrReplaceTempView("ordered_sections")

# Build hierarchy using recursive CTE-like logic with window functions
# We need to track the "most recent heading at each level" as we scan through

hierarchy_sql = """
WITH section_base AS (
    SELECT 
        section_id,
        document_id,
        page_number,
        sequence_number,
        doc_order,
        section_type,
        content_text,
        content_markdown,
        is_strikethrough,
        heading_level,
        bbox_x0,
        bbox_y0,
        bbox_x1,
        bbox_y1,
        extracted_at,
        
        -- Is this a heading?
        CASE 
            WHEN section_type = 'heading' OR heading_level IS NOT NULL THEN TRUE 
            WHEN section_type IN ('list_item', 'paragraph') 
            AND SIZE(SPLIT(content_text, '\\s+')) < 15 THEN TRUE
            ELSE FALSE 
        END as is_heading,
        
        -- Effective heading level (NULL for non-headings, 1-6 for headings)
        CASE 
            WHEN section_type = 'heading' THEN COALESCE(heading_level, 1)
            WHEN heading_level IS NOT NULL THEN heading_level
            ELSE NULL 
        END as effective_level
        
    FROM ordered_sections
),

-- Track the most recent heading at each level using window functions
-- This gives us the "last heading at level N" for any row
heading_tracking AS (
    SELECT 
        *,
        -- Last heading at level 1
        LAST_VALUE(CASE WHEN effective_level = 1 THEN section_id END, TRUE) 
            OVER (PARTITION BY document_id ORDER BY doc_order 
                  ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) as last_h1,
        -- Last heading at level 2
        LAST_VALUE(CASE WHEN effective_level = 2 THEN section_id END, TRUE) 
            OVER (PARTITION BY document_id ORDER BY doc_order 
                  ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) as last_h2,
        -- Last heading at level 3
        LAST_VALUE(CASE WHEN effective_level = 3 THEN section_id END, TRUE) 
            OVER (PARTITION BY document_id ORDER BY doc_order 
                  ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) as last_h3,
        -- Last heading at level 4
        LAST_VALUE(CASE WHEN effective_level = 4 THEN section_id END, TRUE) 
            OVER (PARTITION BY document_id ORDER BY doc_order 
                  ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) as last_h4,
        -- Last heading at level 5
        LAST_VALUE(CASE WHEN effective_level = 5 THEN section_id END, TRUE) 
            OVER (PARTITION BY document_id ORDER BY doc_order 
                  ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) as last_h5,
        -- Last heading at level 6
        LAST_VALUE(CASE WHEN effective_level = 6 THEN section_id END, TRUE) 
            OVER (PARTITION BY document_id ORDER BY doc_order 
                  ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) as last_h6
    FROM section_base
),

-- Determine parent based on hierarchy rules
parent_assignment AS (
    SELECT 
        section_id,
        document_id,
        page_number,
        sequence_number,
        doc_order,
        section_type,
        content_text,
        content_markdown,
        is_strikethrough,
        heading_level,
        effective_level,
        is_heading,
        bbox_x0,
        bbox_y0,
        bbox_x1,
        bbox_y1,
        extracted_at,
        last_h1, last_h2, last_h3, last_h4, last_h5, last_h6,
        
        -- Parent assignment logic:
        -- Headings: parent is the most recent heading at level-1
        -- Non-headings: parent is the most recent heading at any level
        CASE 
            -- Level 1 headings have no parent (root)
            WHEN effective_level = 1 THEN NULL
            -- Level 2 headings → parent is last level 1
            WHEN effective_level = 2 THEN last_h1
            -- Level 3 headings → parent is last level 2
            WHEN effective_level = 3 THEN last_h2
            -- Level 4 headings → parent is last level 3
            WHEN effective_level = 4 THEN last_h3
            -- Level 5 headings → parent is last level 4
            WHEN effective_level = 5 THEN last_h4
            -- Level 6 headings → parent is last level 5
            WHEN effective_level = 6 THEN last_h5
            -- Non-headings → parent is deepest available heading
            ELSE COALESCE(last_h6, last_h5, last_h4, last_h3, last_h2, last_h1)
        END as parent_section_id
        
    FROM heading_tracking
)

SELECT 
    section_id,
    document_id,
    page_number,
    sequence_number,
    doc_order,
    section_type,
    content_text,
    content_markdown,
    is_strikethrough,
    heading_level,
    effective_level,
    is_heading,
    parent_section_id,
    bbox_x0,
    bbox_y0,
    bbox_x1,
    bbox_y1,
    extracted_at
FROM parent_assignment
"""

hierarchy_df = spark.sql(hierarchy_sql)

# Build hierarchy_path (e.g., "1 > 1.3 > 1.3.2")
# This requires iterative path building - we'll do it with a join approach
hierarchy_df.createOrReplaceTempView("sections_with_parent")

# Create hierarchy paths using self-joins (up to 6 levels deep)
hierarchy_path_sql = """
WITH paths AS (
    SELECT 
        s.section_id,
        s.document_id,
        s.page_number,
        s.sequence_number,
        s.doc_order,
        s.section_type,
        s.content_text,
        s.content_markdown,
        s.is_strikethrough,
        s.heading_level,
        s.effective_level,
        s.is_heading,
        s.parent_section_id,
        s.bbox_x0,
        s.bbox_y0,
        s.bbox_x1,
        s.bbox_y1,
        s.extracted_at,
        
        -- Build path by joining to ancestors
        CONCAT_WS(' > ',
            p5.section_id,
            p4.section_id,
            p3.section_id,
            p2.section_id,
            p1.section_id,
            s.section_id
        ) as hierarchy_path_raw
        
    FROM sections_with_parent s
    LEFT JOIN sections_with_parent p1 ON s.parent_section_id = p1.section_id AND s.document_id = p1.document_id
    LEFT JOIN sections_with_parent p2 ON p1.parent_section_id = p2.section_id AND p1.document_id = p2.document_id
    LEFT JOIN sections_with_parent p3 ON p2.parent_section_id = p3.section_id AND p2.document_id = p3.document_id
    LEFT JOIN sections_with_parent p4 ON p3.parent_section_id = p4.section_id AND p3.document_id = p4.document_id
    LEFT JOIN sections_with_parent p5 ON p4.parent_section_id = p5.section_id AND p4.document_id = p5.document_id
)
SELECT 
    section_id,
    document_id,
    page_number,
    sequence_number,
    doc_order,
    section_type,
    content_text,
    content_markdown,
    is_strikethrough,
    heading_level,
    effective_level,
    is_heading,
    parent_section_id,
    -- Clean up the path (remove leading ' > ')
    REGEXP_REPLACE(TRIM(hierarchy_path_raw), '^\\s*>\\s*', '') as hierarchy_path,
    bbox_x0,
    bbox_y0,
    bbox_x1,
    bbox_y1,
    extracted_at
FROM paths
"""

sections_with_hierarchy_df = spark.sql(hierarchy_path_sql)

# Determine is_leaf (sections with no children)
sections_with_hierarchy_df.createOrReplaceTempView("sections_hierarchy")

leaf_detection_sql = """
SELECT 
    s.*,
    CASE 
        WHEN EXISTS (
            SELECT 1 FROM sections_hierarchy c 
            WHERE c.parent_section_id = s.section_id 
              AND c.document_id = s.document_id
        ) THEN FALSE 
        ELSE TRUE 
    END as is_leaf
FROM sections_hierarchy s
"""

hierarchy_complete_df = spark.sql(leaf_detection_sql)

print(f"Sections with hierarchy: {hierarchy_complete_df.count()} rows")
print(f"Headings: {hierarchy_complete_df.filter(F.col('is_heading')).count()}")
print(f"Leaf sections: {hierarchy_complete_df.filter(F.col('is_leaf')).count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC # Step 4: Merge Broken Paragraphs
# MAGIC
# MAGIC Merge consecutive paragraphs that belong together:
# MAGIC - Same page
# MAGIC - Same parent heading
# MAGIC - Each is short (<200 words)

# COMMAND ----------

# Add word count for merge decisions
sections_for_merge_df = hierarchy_complete_df.withColumn(
    "word_count",
    F.size(F.split(F.col("content_text"), "\\s+"))
)

# Register for merge logic
sections_for_merge_df.createOrReplaceTempView("sections_for_merge")

# Identify merge groups: consecutive paragraphs on same page with same parent
merge_groups_sql = f"""
WITH merge_candidates AS (
    SELECT 
        *,
        -- Identify if this row can be merged with previous
        CASE 
            WHEN section_type = 'paragraph' 
             AND word_count <= {config.paragraph_merge_max_words}
             AND LAG(section_type) OVER (
                   PARTITION BY document_id, page_number, COALESCE(parent_section_id, 'ROOT')
                   ORDER BY doc_order
                 ) = 'paragraph'
             AND LAG(word_count) OVER (
                   PARTITION BY document_id, page_number, COALESCE(parent_section_id, 'ROOT')
                   ORDER BY doc_order
                 ) <= {config.paragraph_merge_max_words}
            THEN 1
            ELSE 0
        END as can_merge_with_prev
    FROM sections_for_merge
),

-- Create merge group identifiers
merge_groups AS (
    SELECT 
        *,
        -- Group ID: increments when we can't merge with previous
        SUM(CASE WHEN can_merge_with_prev = 0 THEN 1 ELSE 0 END) 
            OVER (PARTITION BY document_id ORDER BY doc_order) as merge_group_id
    FROM merge_candidates
)

SELECT * FROM merge_groups
"""

sections_with_merge_groups_df = spark.sql(merge_groups_sql)
sections_with_merge_groups_df.createOrReplaceTempView("sections_merge_groups")

# Perform the actual merge: aggregate content within each merge group
merged_sections_sql = """
WITH merged AS (
    SELECT 
        -- Keep first section_id in group
        FIRST_VALUE(section_id) OVER (
            PARTITION BY document_id, merge_group_id 
            ORDER BY doc_order
        ) as section_id,
        document_id,
        -- Keep first page_number (should be same within group)
        MIN(page_number) OVER (PARTITION BY document_id, merge_group_id) as page_number,
        -- Keep first sequence_number
        FIRST_VALUE(sequence_number) OVER (
            PARTITION BY document_id, merge_group_id 
            ORDER BY doc_order
        ) as sequence_number,
        -- Keep first doc_order
        MIN(doc_order) OVER (PARTITION BY document_id, merge_group_id) as doc_order,
        -- Keep section_type from first (they should all be 'paragraph' if merging)
        FIRST_VALUE(section_type) OVER (
            PARTITION BY document_id, merge_group_id 
            ORDER BY doc_order
        ) as section_type,
        -- Keep first for hierarchy-related fields
        FIRST_VALUE(heading_level) OVER (
            PARTITION BY document_id, merge_group_id 
            ORDER BY doc_order
        ) as heading_level,
        FIRST_VALUE(effective_level) OVER (
            PARTITION BY document_id, merge_group_id 
            ORDER BY doc_order
        ) as effective_level,
        FIRST_VALUE(is_heading) OVER (
            PARTITION BY document_id, merge_group_id 
            ORDER BY doc_order
        ) as is_heading,
        FIRST_VALUE(parent_section_id) OVER (
            PARTITION BY document_id, merge_group_id 
            ORDER BY doc_order
        ) as parent_section_id,
        FIRST_VALUE(hierarchy_path) OVER (
            PARTITION BY document_id, merge_group_id 
            ORDER BY doc_order
        ) as hierarchy_path,
        FIRST_VALUE(is_leaf) OVER (
            PARTITION BY document_id, merge_group_id 
            ORDER BY doc_order
        ) as is_leaf,
        FIRST_VALUE(is_strikethrough) OVER (
            PARTITION BY document_id, merge_group_id 
            ORDER BY doc_order
        ) as is_strikethrough,
        FIRST_VALUE(extracted_at) OVER (
            PARTITION BY document_id, merge_group_id 
            ORDER BY doc_order
        ) as extracted_at,
        -- Bbox from first
        FIRST_VALUE(bbox_x0) OVER (PARTITION BY document_id, merge_group_id ORDER BY doc_order) as bbox_x0,
        FIRST_VALUE(bbox_y0) OVER (PARTITION BY document_id, merge_group_id ORDER BY doc_order) as bbox_y0,
        FIRST_VALUE(bbox_x1) OVER (PARTITION BY document_id, merge_group_id ORDER BY doc_order) as bbox_x1,
        FIRST_VALUE(bbox_y1) OVER (PARTITION BY document_id, merge_group_id ORDER BY doc_order) as bbox_y1,
        
        merge_group_id,
        content_text,
        content_markdown,
        doc_order as original_order,
        
        -- Count items in this merge group
        COUNT(*) OVER (PARTITION BY document_id, merge_group_id) as items_in_group
        
    FROM sections_merge_groups
),

-- Now aggregate the text within merge groups
aggregated AS (
    SELECT 
        section_id,
        document_id,
        page_number,
        sequence_number,
        doc_order,
        section_type,
        heading_level,
        effective_level,
        is_heading,
        parent_section_id,
        hierarchy_path,
        is_leaf,
        is_strikethrough,
        extracted_at,
        bbox_x0,
        bbox_y0,
        bbox_x1,
        bbox_y1,
        merge_group_id,
        items_in_group,
        
        -- Concatenate all content_text in this group
        CONCAT_WS(' ', COLLECT_LIST(content_text) OVER (
            PARTITION BY document_id, merge_group_id 
            ORDER BY original_order
            ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
        )) as merged_content_text,
        
        -- Concatenate all content_markdown in this group
        CONCAT_WS('\n\n', COLLECT_LIST(content_markdown) OVER (
            PARTITION BY document_id, merge_group_id 
            ORDER BY original_order
            ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
        )) as merged_content_markdown,
        
        -- Row number within group to keep only first
        ROW_NUMBER() OVER (PARTITION BY document_id, merge_group_id ORDER BY original_order) as rn
        
    FROM merged
)

-- Keep only the first row of each merge group (with aggregated content)
SELECT 
    section_id,
    document_id,
    page_number,
    sequence_number,
    doc_order,
    section_type,
    merged_content_text as content_text,
    merged_content_markdown as content_markdown,
    heading_level,
    effective_level,
    is_heading,
    parent_section_id,
    hierarchy_path,
    is_leaf,
    is_strikethrough,
    extracted_at,
    bbox_x0,
    bbox_y0,
    bbox_x1,
    bbox_y1,
    items_in_group as merged_count
FROM aggregated
WHERE rn = 1
"""

merged_sections_df = spark.sql(merged_sections_sql)

print(f"Sections after merge: {merged_sections_df.count()} rows")
print(f"Reduction from merge: {sections_with_merge_groups_df.count() - merged_sections_df.count()} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC # Step 5: Semantic Chunking
# MAGIC
# MAGIC Split long narrative sections (>200 words) into smaller chunks.
# MAGIC - Only chunk long narrative sections (paragraphs)
# MAGIC - Split by sentence boundaries
# MAGIC - Create new rows with section_id + "_c1", "_c2", etc.

# COMMAND ----------

from pyspark.sql.types import ArrayType, StructType, StructField, StringType, IntegerType

# UDF-free approach: use SQL string functions for sentence splitting
# We'll use TRANSFORM and FILTER for clean Spark SQL implementation

merged_sections_df.createOrReplaceTempView("merged_sections")

# Add word count and identify sections needing chunking
chunking_prep_sql = f"""
SELECT 
    *,
    SIZE(SPLIT(content_text, '\\\\s+')) as word_count,
    -- Sections to chunk: paragraphs with word_count > threshold
    CASE 
        WHEN section_type = 'paragraph' 
         AND SIZE(SPLIT(content_text, '\\\\s+')) > {config.chunk_word_threshold}
        THEN TRUE 
        ELSE FALSE 
    END as needs_chunking
FROM merged_sections
"""

sections_prep_df = spark.sql(chunking_prep_sql)
sections_prep_df.createOrReplaceTempView("sections_prep")

# For sections that don't need chunking, pass through as-is
no_chunk_df = spark.sql("""
    SELECT 
        section_id,
        document_id,
        page_number,
        sequence_number,
        doc_order,
        section_type,
        content_text,
        content_markdown,
        heading_level,
        effective_level,
        is_heading,
        parent_section_id,
        hierarchy_path,
        is_leaf,
        is_strikethrough,
        extracted_at,
        bbox_x0,
        bbox_y0,
        bbox_x1,
        bbox_y1,
        merged_count,
        word_count,
        NULL as chunk_index
    FROM sections_prep
    WHERE needs_chunking = FALSE
""")

# For sections that need chunking, we need to split by sentences
# Split on sentence boundaries: ". ", "! ", "? " followed by uppercase or newline
# Then group sentences into chunks of ~150 words each

# First get the sections that need chunking
needs_chunking_df = spark.sql("""
    SELECT * FROM sections_prep WHERE needs_chunking = TRUE
""").collect()

# Process chunking in Python (then convert back to Spark)
# This is more reliable for sentence boundary detection than pure SQL
import re

def chunk_text(text, target_words=150):
    """Split text into chunks of approximately target_words."""
    if not text:
        return [("", 1)]
    
    # Split on sentence boundaries
    # Pattern: period/exclamation/question followed by space and uppercase, or newline
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])\s*\n+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if not sentences:
        return [(text, 1)]
    
    chunks = []
    current_chunk = []
    current_words = 0
    
    for sentence in sentences:
        sentence_words = len(sentence.split())
        
        if current_words + sentence_words > target_words and current_chunk:
            # Save current chunk and start new one
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_words = sentence_words
        else:
            current_chunk.append(sentence)
            current_words += sentence_words
    
    # Don't forget the last chunk
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    # Return with chunk indices (1-based)
    return [(chunk, idx + 1) for idx, chunk in enumerate(chunks)]


# Create chunked rows
chunked_rows = []
for row in needs_chunking_df:
    text_chunks = chunk_text(row['content_text'], config.chunk_target_words)
    md_chunks = chunk_text(row['content_markdown'], config.chunk_target_words)
    
    # Ensure same number of chunks for text and markdown
    max_chunks = max(len(text_chunks), len(md_chunks))
    
    for i in range(max_chunks):
        chunk_text_content = text_chunks[i][0] if i < len(text_chunks) else ""
        chunk_md_content = md_chunks[i][0] if i < len(md_chunks) else ""
        chunk_idx = i + 1
        
        chunked_rows.append({
            "section_id": f"{row['section_id']}_c{chunk_idx}",
            "document_id": row['document_id'],
            "page_number": row['page_number'],
            "sequence_number": row['sequence_number'],
            "doc_order": row['doc_order'],
            "section_type": row['section_type'],
            "content_text": chunk_text_content,
            "content_markdown": chunk_md_content,
            "heading_level": row['heading_level'],
            "effective_level": row['effective_level'],
            "is_heading": row['is_heading'],
            "parent_section_id": row['parent_section_id'],
            "hierarchy_path": row['hierarchy_path'],
            "is_leaf": row['is_leaf'],
            "is_strikethrough": row['is_strikethrough'],
            "extracted_at": row['extracted_at'],
            "bbox_x0": row['bbox_x0'],
            "bbox_y0": row['bbox_y0'],
            "bbox_x1": row['bbox_x1'],
            "bbox_y1": row['bbox_y1'],
            "merged_count": row['merged_count'],
            "word_count": len(chunk_text_content.split()),
            "chunk_index": chunk_idx
        })

# Create DataFrame from chunked rows
if chunked_rows:
    chunk_schema = StructType([
        StructField("section_id", StringType(), True),
        StructField("document_id", StringType(), True),
        StructField("page_number", IntegerType(), True),
        StructField("sequence_number", IntegerType(), True),
        StructField("doc_order", IntegerType(), True),
        StructField("section_type", StringType(), True),
        StructField("content_text", StringType(), True),
        StructField("content_markdown", StringType(), True),
        StructField("heading_level", IntegerType(), True),
        StructField("effective_level", IntegerType(), True),
        StructField("is_heading", BooleanType(), True),
        StructField("parent_section_id", StringType(), True),
        StructField("hierarchy_path", StringType(), True),
        StructField("is_leaf", BooleanType(), True),
        StructField("is_strikethrough", BooleanType(), True),
        StructField("extracted_at", StringType(), True),
        StructField("bbox_x0", DoubleType(), True),
        StructField("bbox_y0", DoubleType(), True),
        StructField("bbox_x1", DoubleType(), True),
        StructField("bbox_y1", DoubleType(), True),
        StructField("merged_count", IntegerType(), True),
        StructField("word_count", IntegerType(), True),
        StructField("chunk_index", IntegerType(), True),
    ])
    
    chunked_df = spark.createDataFrame(chunked_rows, schema=chunk_schema)
else:
    # Empty DataFrame with same schema
    chunked_df = no_chunk_df.limit(0)

# Combine non-chunked and chunked sections
all_sections_df = no_chunk_df.unionByName(chunked_df)

print(f"Sections after chunking: {all_sections_df.count()} rows")
print(f"Original long sections chunked: {len(needs_chunking_df)}")
print(f"New chunk rows created: {chunked_df.count() if chunked_rows else 0}")

# COMMAND ----------

# MAGIC %md
# MAGIC # Step 6: Add Structural Columns & Final Ordering

# COMMAND ----------

# Register for final processing
all_sections_df.createOrReplaceTempView("all_sections")

# Final Silver DataFrame with all required columns
silver_sql = """
SELECT 
    section_id,
    document_id,
    page_number,
    
    -- Global order post-merge + chunk (deterministic ordering)
    ROW_NUMBER() OVER (
        PARTITION BY document_id 
        ORDER BY page_number, sequence_number, COALESCE(chunk_index, 0)
    ) as section_order,
    
    section_type,
    content_text,
    content_markdown,
    heading_level,
    parent_section_id,
    hierarchy_path,
    chunk_index,
    
    -- Content metrics
    LENGTH(content_text) as content_length,
    SIZE(SPLIT(content_text, '\\s+')) as word_count,
    
    -- Classification flags
    is_heading,
    is_leaf,
    is_strikethrough,
    
    -- Metadata
    extracted_at,
    CURRENT_TIMESTAMP() as silver_processed_at
    
FROM all_sections
"""

silver_sections_df = spark.sql(silver_sql)

print(f"Final Silver sections: {silver_sections_df.count()} rows")
silver_sections_df.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC # Step 7: Write Silver Table

# COMMAND ----------

silver_table = config.full_table(config.silver_sections)
from delta.tables import DeltaTable #indempotency

silver_delta = DeltaTable.forName(spark, silver_table)

(
    silver_delta.alias("target")
    .merge(
        silver_sections_df.alias("source"),
        """
        target.document_id = source.document_id AND
        target.section_id = source.section_id AND
        (
            target.chunk_index = source.chunk_index OR
            (target.chunk_index IS NULL AND source.chunk_index IS NULL)
        )
        """
    )
    .whenMatchedUpdate(
    set = {
        "content_text": "source.content_text",
        "content_markdown": "source.content_markdown",
        "heading_level": "source.heading_level",
        "parent_section_id": "source.parent_section_id",
        "hierarchy_path": "source.hierarchy_path",
        "chunk_index": "source.chunk_index",
        "content_length": "source.content_length",
        "word_count": "source.word_count",
        "is_heading": "source.is_heading",
        "is_leaf": "source.is_leaf",
        "is_strikethrough": "source.is_strikethrough",
        "extracted_at": "source.extracted_at",
        "silver_processed_at": "source.silver_processed_at"
    }
)
    .whenNotMatchedInsert(
    values = {
        "document_id": "source.document_id",
        "section_id": "source.section_id",
        "page_number": "source.page_number",
        "section_order": "source.section_order",
        "section_type": "source.section_type",
        "content_text": "source.content_text",
        "content_markdown": "source.content_markdown",
        "heading_level": "source.heading_level",
        "parent_section_id": "source.parent_section_id",
        "hierarchy_path": "source.hierarchy_path",
        "chunk_index": "source.chunk_index",
        "content_length": "source.content_length",
        "word_count": "source.word_count",
        "is_heading": "source.is_heading",
        "is_leaf": "source.is_leaf",
        "is_strikethrough": "source.is_strikethrough",
        "extracted_at": "source.extracted_at",
        "silver_processed_at": "source.silver_processed_at"
    }
)
    .execute()
)

last_doc_id = silver_sections_df.select("document_id").first()["document_id"]
print(f"Silver MERGE complete for document: {last_doc_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC # Validation & Summary

# COMMAND ----------

print("\nSECTION TYPE DISTRIBUTION")
display(spark.sql(f"""
    SELECT 
        section_type,
        COUNT(*) AS count,
        ROUND(AVG(word_count), 1) AS avg_words,
        ROUND(AVG(content_length), 0) AS avg_chars
    FROM {silver_table}
    WHERE chunk_index IS NULL
    GROUP BY section_type
    ORDER BY count DESC
"""))


print("\nHIERARCHY DEPTH")
display(spark.sql(f"""
    SELECT 
        SIZE(SPLIT(hierarchy_path, ' > ')) AS depth,
        COUNT(*) AS sections
    FROM {silver_table}
    WHERE chunk_index IS NULL
      AND hierarchy_path IS NOT NULL
      AND hierarchy_path != ''
    GROUP BY SIZE(SPLIT(hierarchy_path, ' > '))
    ORDER BY depth
"""))


print("\nCHUNKING SUMMARY")
display(spark.sql(f"""
    SELECT 
        CASE WHEN chunk_index IS NULL THEN 'Not Chunked' ELSE 'Chunked' END AS chunking_status,
        COUNT(*) AS sections,
        ROUND(AVG(word_count), 1) AS avg_words
    FROM {silver_table}
    GROUP BY CASE WHEN chunk_index IS NULL THEN 'Not Chunked' ELSE 'Chunked' END
"""))


print("\nPER-DOCUMENT SUMMARY")
display(spark.sql(f"""
    SELECT 
        document_id,
        COUNT(CASE WHEN chunk_index IS NULL THEN 1 END) AS total_sections,
        SUM(CASE WHEN is_heading AND chunk_index IS NULL THEN 1 ELSE 0 END) AS headings,
        SUM(CASE WHEN is_leaf AND chunk_index IS NULL THEN 1 ELSE 0 END) AS leaf_sections,
        SUM(CASE WHEN chunk_index IS NOT NULL THEN 1 ELSE 0 END) AS chunks,
        ROUND(SUM(word_count), 0) AS total_words
    FROM {silver_table}
    GROUP BY document_id
"""))


# COMMAND ----------

# MAGIC %md
# MAGIC # Sample Output

# COMMAND ----------

# Show sample Silver sections
print("\nSAMPLE SILVER SECTIONS (first 10)")
display(spark.sql(f"""
    SELECT 
        section_id,
        document_id,
        page_number,
        section_order,
        section_type,
        SUBSTRING(content_text, 1, 100) as content_preview,
        heading_level,
        parent_section_id,
        chunk_index,
        word_count,
        is_heading,
        is_leaf
    FROM {silver_table}
    ORDER BY document_id, section_order
    LIMIT 10
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Output Schema:
# MAGIC
# MAGIC ```
# MAGIC section_id          - Original or chunked ID (e.g., doc_s00001 or doc_s00001_c1)
# MAGIC document_id         - Source document
# MAGIC page_number         - Page location
# MAGIC section_order       - Global order within document
# MAGIC section_type        - heading, paragraph, list_item, etc.
# MAGIC content_text        - Clean text content
# MAGIC content_markdown    - Markdown formatted content
# MAGIC heading_level       - 1-6 for headings, NULL otherwise
# MAGIC parent_section_id   - Hierarchical parent
# MAGIC hierarchy_path      - Full path (e.g., "s00001 > s00005 > s00012")
# MAGIC chunk_index         - Chunk number if chunked, NULL otherwise
# MAGIC content_length      - Character count
# MAGIC word_count          - Word count
# MAGIC is_heading          - Boolean flag
# MAGIC is_leaf             - Boolean flag (no children)
# MAGIC is_strikethrough    - Boolean flag
# MAGIC extracted_at        - Original extraction timestamp
# MAGIC silver_processed_at - This pipeline timestamp
# MAGIC ```