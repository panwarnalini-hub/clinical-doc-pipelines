# Databricks notebook source
# MAGIC %sql
# MAGIC -- 1. CHECK BRONZE EXTRACTION
# MAGIC SELECT * FROM dev_clinical.doc_test.bronze_sections ORDER BY 1 LIMIT 20;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Check extracted tables
# MAGIC SELECT *
# MAGIC FROM dev_clinical.doc_test.bronze_tables
# MAGIC LIMIT 50;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- 2. CHECK SILVER SECTIONS
# MAGIC SELECT document_id, section_id, section_type, content_text, is_heading, word_count
# MAGIC FROM dev_clinical.doc_test.silver_sections
# MAGIC ORDER BY document_id, section_order
# MAGIC LIMIT 50;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- 3. CHECK EMBEDDINGS
# MAGIC SELECT section_id, embedding_type, model_name, vector_length
# MAGIC FROM dev_clinical.doc_test.silver_section_embeddings
# MAGIC LIMIT 20;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- 4. CHECK ALL CLASSIFICATIONS (not just specific ones)
# MAGIC SELECT classification, classification_status, COUNT(*) as count
# MAGIC FROM dev_clinical.doc_test.silver_section_classifications
# MAGIC GROUP BY classification, classification_status
# MAGIC ORDER BY count DESC;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- 5. SEE ACTUAL CLASSIFICATIONS WITH TITLES
# MAGIC SELECT section_title, classification, classification_confidence, classification_status
# MAGIC FROM dev_clinical.doc_test.silver_section_classifications
# MAGIC ORDER BY classification_confidence DESC
# MAGIC LIMIT 30;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- 6. CHECK WHAT GOT UNMAPPED (and why)
# MAGIC SELECT section_title, top_k_labels[0] as best_guess, top_k_scores[0] as best_score
# MAGIC FROM dev_clinical.doc_test.silver_section_classifications
# MAGIC WHERE classification_status = 'UNMAPPED'
# MAGIC ORDER BY top_k_scores[0] DESC;
