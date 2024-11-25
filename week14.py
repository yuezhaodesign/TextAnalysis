import streamlit as st
import spacy_streamlit # this is the spacy streamlit library

spacy_model = "en_core_web_sm"

# Create the title and text area for the user to input text
st.title("Text Analysis App")
DEFAULT_TEXT = """In 1749, Benjamin Franklin—printer, inventor, and future founding father of the United States—published his famous essay, “Proposals Relating to the Education of Youth,” circulated it among Philadelphia’s leading citizens, and organized 24 trustees to form an institution of higher education based on his proposals. The group purchased the building and in 1751, opened its doors to children of the gentry and working class alike as the Academy and Charitable School in the Province of Pennsylvania. Franklin served as president of the institution until 1755 and continued to serve as a trustee until his death in 1790.
"""
text = st.text_area("Text to analyze", DEFAULT_TEXT, height=220)


# Process the text
doc = spacy_streamlit.process_text(spacy_model, text)

# Visualize Named Entity Recognition(NER)
spacy_streamlit.visualize_ner(
    doc,
    labels=["PERSON", "DATE", "TIME", "GPE", "ORG", "FAC", "PRODUCT", "EVENT", "WORK_OF_ART", "LAW", "LANGUAGE", "LOC", "MONEY", "NORP", "ORDINAL", "PERCENT", "QUANTITY", "CARDINAL"],
    show_table=False,
    title="Named Entity Recognition Highlight",
)

# Tag Part of Speech
spacy_streamlit.visualize_tokens(
    doc,
    attrs=["text", "pos_", "dep_", "lemma_", "is_alpha", "is_stop"],
    # pos_ = Part of Speech, dep_ = Dependency, lemma_ = Lemma, is_alpha = Alphabetical, is_stop = Stop Word
    title='Tokens Analysis',
)

# Visualize the dependency tree
spacy_streamlit.visualize_parser(
    doc,
    title="Dependency Parse",
    )



