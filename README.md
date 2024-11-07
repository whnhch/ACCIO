# ACCIO: Table Understanding Enhanced via Contrastive Learning with Aggregations
TLDR; Contrastive Learning between tabular data and its aggregation, Pivot Tables for table understanding.

The attention to table understanding using recent natural language models has
  been growing. However, most related works tend to focus on learning the
  structure of the table directly. Just as humans improve their understanding of
  sentences by comparing them, they can also enhance their understanding by
  comparing tables. With this idea, in this paper, we introduce **ACCIO**, t**A**ble understanding enhan**C**ed via **C**ontrastive
  learn**I**ng with aggregati**O**ns, a novel approach to enhancing
  table understanding by contrasting original tables with their pivot summaries
  through contrastive learning. ACCIO trains an encoder to bring these table
  pairs closer together. Through validation via column type annotation, ACCIO
  achieves competitive performance with a macro F1 score of 91.1 compared to
  state-of-the-art methods. This work represents the first attempt to utilize
  pairs of tables for table embedding, promising significant advancements in
  table comprehension.
