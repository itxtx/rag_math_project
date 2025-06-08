# src/data_ingestion/latex_parser.py
import os
import re
import uuid
import pickle
import networkx as nx
from sentence_transformers import SentenceTransformer
import xml.etree.ElementTree as ET

# This is our robust, low-level processor
from . import latex_processor

class LatexToGraphParser:
    """
    Parses a LaTeX document by converting it to XML and then extracts structured
    nodes (theorems, definitions, etc.) and their relationships to build a knowledge graph.
    """
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.embedding_model = SentenceTransformer(model_name)
        self.graph = nx.DiGraph()

    def _get_clean_text(self, element) -> str:
        """Extracts clean, readable text from an XML element."""
        if element is None:
            return ""
        text_chunks = [text.strip() for text in element.itertext() if text.strip()]
        return "\n\n".join(text_chunks)

    def extract_structured_nodes(self, latex_content: str, doc_id: str, source: str = None):
        """
        Extracts environments like theorem, definition, etc., from a single XML tree.
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

        environments_to_find = [
            'theorem', 'lemma', 'proposition', 'corollary', 'definition',
            'example', 'remark', 'proof'
        ]

        for env_name in environments_to_find:
            env_count = 0
            for element in root.findall(f".//{env_name}"):
                env_count += 1
                env_content_clean = self._get_clean_text(element)
                if not env_content_clean:
                    continue

                label = element.get('id', f"{env_name}-{uuid.uuid4().hex[:8]}")
                refs = [ref.get('refid') for ref in element.findall('.//ref') if ref.get('refid')]
                embedding = self.embedding_model.encode(env_content_clean, convert_to_tensor=False)

                self.graph.add_node(
                    label,
                    node_type=env_name,
                    doc_id=doc_id,
                    source=source,
                    text=env_content_clean,
                    embedding=embedding
                )

                for ref_label in refs:
                    self.graph.add_edge(label, ref_label, edge_type='references')

            if env_count > 0:
                print(f"Found {env_count} {env_name} environments in {doc_id}")

        if not self.graph.nodes:
             print(f"Skipping LaTeX file due to no structured content found: {source or doc_id}")

        print(f"Total nodes in graph after processing {doc_id}: {len(self.graph.nodes)}")
        print(f"Total edges in graph after processing {doc_id}: {len(self.graph.edges)}")

    def save_graph_and_embeddings(self, graph_path, embeddings_path):
        """Saves the final graph and initial embeddings."""
        print(f"Saving knowledge graph to {graph_path}")
        
        # Create a copy of the graph for saving
        save_graph = nx.DiGraph()
        
        # Copy nodes and edges, removing embedding data
        for node, data in self.graph.nodes(data=True):
            node_data = data.copy()
            # Remove embedding from graph data
            if 'embedding' in node_data:
                del node_data['embedding']
            save_graph.add_node(node, **node_data)
        
        # Copy edges
        for u, v, data in self.graph.edges(data=True):
            save_graph.add_edge(u, v, **data)
        
        # Save the modified graph
        nx.write_graphml(save_graph, graph_path)

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