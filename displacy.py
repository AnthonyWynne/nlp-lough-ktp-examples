# %%
import spacy

nlp = spacy.load('en_core_web_sm')
text = 'Born in Medellín, Spain, to a family of lesser nobility, Cortés chose to pursue adventure and riches in the New World. He went to Hispaniola and later to Cuba, where he received an encomienda (the right to the labor of certain subjects). For a short time, he served as alcalde (magistrate) of the second Spanish town founded on the island. In 1519, he was elected captain of the third expedition to the mainland, which he partly funded. His enmity with the Governor of Cuba, Diego Velázquez de Cuéllar, resulted in the recall of the expedition at the last moment, an order which Cortés ignored. '
doc = nlp(text)
# Generate the tree visualization for the sentence with all the grammatical functions
options = {
    'compact': True,
    'color': 'yellow',
    'bg': '#272822',
    'font': 'Source Sans Pro'}

spacy.displacy.render(doc, style='dep', jupyter=True, options=options)