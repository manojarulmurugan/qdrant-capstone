# Test queries

Five natural-language queries, none of which quote any document verbatim. Each one is aimed at a different newsgroup so that a correct result is easy to eyeball from the `category` payload.

| # | Query | Categories I expect to see |
|---|-------|----------------------------|
| 1 | a question about a graphics card driver | `comp.graphics`, `comp.sys.ibm.pc.hardware`, `comp.os.ms-windows.misc` |
| 2 | what does the Bible say about the resurrection | `soc.religion.christian`, `talk.religion.misc`, `alt.atheism` |
| 3 | for sale: used motorcycle in good condition | `misc.forsale`, `rec.motorcycles` |
| 4 | government encryption policy and the clipper chip | `sci.crypt`, `talk.politics.misc` |
| 5 | treatment options for chronic back pain | `sci.med` |

The literal strings live in `TEST_QUERIES` in `embed.py`, so the file that embeds them is the single source of truth and this table cannot drift out of sync silently.
