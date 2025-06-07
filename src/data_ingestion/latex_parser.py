# src/data_ingestion/latex_parser.py
import os
import re
import uuid
import pickle
import networkx as nx
from pylatexenc.latexwalker import LatexWalker, LatexEnvironmentNode, LatexMacroNode
from sentence_transformers import SentenceTransformer

# This is our new, more robust processor
from . import latex_processor

class LatexToGraphParser:
    """
    Parses a pre-processed LaTeX document to extract structured nodes 
    (theorems, definitions, etc.) and their relationships, building a knowledge graph.
    """
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.embedding_model = SentenceTransformer(model_name)
        # We'll build the graph as we parse documents
        self.graph = nx.DiGraph() 

    def extract_structured_nodes(self, latex_content: str, doc_id: str):
        """
        Extracts environments like theorem, definition, etc., as nodes for the graph.
        """
        # First, pre-process the entire document body with LaTeXML
        # This gives us clean text with standard math notation.
        clean_content = latex_processor.process_latex_document(latex_content)

        if not clean_content:
            print(f"WARNING: LaTeXML returned no content for doc_id: {doc_id}")
            return

        # Now, use pylatexenc to find the structure (environments)
        # We run it on the *original* content to find the structure, but use the
        # *clean* content for the text. This is a hybrid approach.
        lw = LatexWalker(latex_content)
        nodes, _, _ = lw.get_latex_nodes()

        for node in nodes:
            if node.isNodeType(LatexEnvironmentNode):
                env_name = node.environmentname
                # Define which environments we care about
                if env_name in ['theorem', 'lemma', 'definition', 'proof', 'corollary']:
                    # Get the raw text of the environment to process it separately
                    env_content_raw = node.nodelist.latex_verbatim()
                    # Get the clean text from LaTeXML for this specific snippet
                    env_content_clean = latex_processor.process_latex_document(env_content_raw)
                    
                    if not env_content_clean:
                        continue

                    # Find label for ID, and refs for edges
                    label = self._find_label(node.nodelist) or f"{env_name}-{uuid.uuid4().hex[:8]}"
                    refs = self._find_refs(node.nodelist)

                    # Generate the initial text embedding
                    embedding = self.embedding_model.encode(env_content_clean, convert_to_tensor=False)

                    # Add the node to the graph
                    self.graph.add_node(
                        label,
                        node_type=env_name,
                        doc_id=doc_id,
                        text=env_content_clean,
                        embedding=embedding
                    )
                    
                    # Add edges for each reference found
                    for ref_label in refs:
                        # We add the edge now, even if the 'ref_label' node doesn't exist yet.
                        # It will be created when its own environment is parsed.
                        self.graph.add_edge(label, ref_label, edge_type='references')

    def _find_label(self, nodelist):
        for node in nodelist:
            if node.isNodeType(LatexMacroNode) and node.macroname == 'label':
                return node.nodeargs[0].nodelist.latex_verbatim()
        return None

    def _find_refs(self, nodelist):
        refs = []
        for node in nodelist:
            if node.isNodeType(LatexMacroNode) and node.macroname in ['ref', 'eqref']:
                refs.append(node.nodeargs[0].nodelist.latex_verbatim())
        return refs

    def save_graph_and_embeddings(self, graph_path, embeddings_path):
        """Saves the final graph and initial embeddings."""
        print(f"Saving knowledge graph to {graph_path}")
        nx.write_graphml(self.graph, graph_path)

        initial_embeddings = {node: data['embedding'] for node, data in self.graph.nodes(data=True)}
        print(f"Saving initial text embeddings to {embeddings_path}")
        with open(embeddings_path, 'wb') as f:
            pickle.dump(initial_embeddings, f)
