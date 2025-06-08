# src/data_ingestion/latex_parser.py
import os
import re
import uuid
import pickle
import networkx as nx
from sentence_transformers import SentenceTransformer
import xml.etree.ElementTree as ET

from . import latex_processor

class LatexToGraphParser:
    """
    Parses a LaTeX document by converting it to XML and then extracts structured
    nodes (theorems, definitions, etc.) and their relationships to build a knowledge graph.
    """
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.embedding_model = SentenceTransformer(model_name)
        self.graph = nx.DiGraph()

    def _get_node_type_from_title(self, title_text: str) -> str:
        """Determines the environment type from its title text."""
        title_lower = title_text.lower()
        # Order matters: 'proposition' should be checked before 'proof' if titles can be "Proof of Proposition..."
        for env_type in ['theorem', 'lemma', 'proposition', 'corollary', 'definition', 'example', 'remark', 'proof']:
            if title_lower.startswith(env_type):
                return env_type
        return 'theorem' # Default if no specific type is found in the title

    def _get_main_text_from_element(self, element) -> str:
        """Extracts text ONLY from the main content paragraphs (<para> tags) to avoid duplication."""
        if element is None:
            return ""
        # Find all <para> children and join their text. This is more robust.
        paras = element.findall('.//para')
        if not paras: # Fallback for structures that might not use <para>
            return "\n\n".join(text.strip() for text in element.itertext() if text.strip())
            
        text_chunks = ["".join(p.itertext()).strip() for p in paras]
        return "\n\n".join(chunk for chunk in text_chunks if chunk)

    def extract_structured_nodes(self, latex_content: str, doc_id: str, source: str = None):
        """
        Extracts environments like theorem, definition, etc., by intelligently parsing the XML tree.
        """
        xml_output = latex_processor.run_latexml_on_content(latex_content)

        if not xml_output:
            print(f"WARNING: LaTeXML returned no content for doc_id: {doc_id}. Skipping.")
            return

        xml_output = re.sub(r' xmlns="[^"]+"', '', xml_output, count=1)

        try:
            root = ET.fromstring(xml_output)
        except ET.ParseError as e:
            print(f"FATAL: Could not parse XML for {doc_id}. Error: {e}")
            return

        # Find all theorem-like environments. LaTeXML groups them under the <theorem> tag.
        # This is more efficient than searching for each type individually.
        theorem_elements = root.findall(".//theorem")
        print(f"Found {len(theorem_elements)} theorem-like environments in {doc_id}")

        for element in theorem_elements:
            title_element = element.find('title')
            title_text = "".join(title_element.itertext()).strip() if title_element is not None else ""
            
            # Determine the node type from the title text
            node_type = self._get_node_type_from_title(title_text)
            
            # Extract main content text to avoid duplication
            main_text = self._get_main_text_from_element(element)

            if not main_text.strip():
                continue

            label = element.get('id', f"{node_type}-{uuid.uuid4().hex[:8]}")
            refs = [ref.get('refid') for ref in element.findall('.//ref') if ref.get('refid')]
            embedding = self.embedding_model.encode(main_text, convert_to_tensor=False)

            self.graph.add_node(
                label,
                node_type=node_type,
                doc_id=doc_id,
                source=source,
                text=main_text,
                embedding=embedding
            )

            for ref_label in refs:
                self.graph.add_edge(label, ref_label, edge_type='references')

        # You can add separate logic for 'proofs' if they are tagged differently by LaTeXML
        # For now, this logic assumes proofs are also defined via \newtheorem
        
        if not self.graph.nodes:
             print(f"Skipping LaTeX file due to no structured content found: {source or doc_id}")

        print(f"Total nodes in graph after processing {doc_id}: {len(self.graph.nodes)}")

    def save_graph_and_embeddings(self, graph_path, embeddings_path):
        """Saves the final graph and initial embeddings."""
        print(f"Saving knowledge graph to {graph_path}")
        
        # Create a copy of the graph for saving
        graph_for_saving = self.graph.copy()
        
        # Remove embedding data from nodes before saving to GraphML
        for node in graph_for_saving.nodes():
            if 'embedding' in graph_for_saving.nodes[node]:
                del graph_for_saving.nodes[node]['embedding']
        
        # Save the graph without embeddings
        nx.write_graphml(graph_for_saving, graph_path)

        # Save embeddings separately
        initial_embeddings = {node: data['embedding'] for node, data in self.graph.nodes(data=True) if 'embedding' in data}
        print(f"Saving initial text embeddings to {embeddings_path}")
        with open(embeddings_path, 'wb') as f:
            pickle.dump(initial_embeddings, f)

    def get_graph_nodes_as_conceptual_blocks(self):
        """Returns all nodes from the graph as conceptual blocks."""
        blocks = []
        for node, data in self.graph.nodes(data=True):
            blocks.append({
                'id': node,
                'type': data.get('node_type', 'unknown'),
                'text': data.get('text', ''),
                'doc_id': data.get('doc_id', ''),
                'source': data.get('source', ''),
                'embedding': data.get('embedding', None)
            })
        return blocks