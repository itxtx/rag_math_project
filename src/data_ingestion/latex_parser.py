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

    def _get_clean_text(self, element, exclude_title=True) -> str:
        """
        Extracts text from an XML element, preserving LaTeX math mode content.
        
        Args:
            element: XML element to extract text from
            exclude_title: If True, skip the title element to avoid duplication
        """
        if element is None:
            return ""
        
        # Track seen text to avoid duplicates
        seen_text = set()
        text_chunks = []
        
        def extract_text_recursive(elem, skip_title=False):
            """Recursively extract text while avoiding duplicates"""
            # Skip title elements if requested
            if skip_title and elem.tag == 'title':
                return
            
            # Add the element's text if it exists and isn't just whitespace
            if elem.text and elem.text.strip():
                text = elem.text.strip()
                if text not in seen_text:
                    seen_text.add(text)
                    text_chunks.append(text)
            
            # Process all child elements
            for child in elem:
                # Skip title if we're excluding it
                if exclude_title and child.tag == 'title':
                    continue
                    
                # Handle math elements specially
                if child.tag == 'Math':
                    tex = child.get('tex')
                    if tex and tex not in seen_text:
                        seen_text.add(tex)
                        # Wrap in $ to preserve math mode
                        text_chunks.append(f"${tex}$")
                else:
                    # Recursively process other elements
                    extract_text_recursive(child, skip_title=False)
                
                # Add the tail text (text after the element)
                if child.tail and child.tail.strip():
                    tail_text = child.tail.strip()
                    if tail_text not in seen_text:
                        seen_text.add(tail_text)
                        text_chunks.append(tail_text)
        
        # Start extraction
        extract_text_recursive(element, skip_title=exclude_title)
        
        # Join all text chunks with spaces
        full_text = " ".join(text_chunks)
        
        # Clean up any repeated patterns that might have slipped through
        # Remove patterns like "Proof 12 12 Proof 12 Proof"
        full_text = re.sub(r'(\b\w+\b)(\s+\d+)*(\s+\1)+', r'\1', full_text)
        
        # Remove standalone numbers that might be labels
        full_text = re.sub(r'\b\d+\b\s*', '', full_text)
        
        # Clean up multiple spaces
        full_text = re.sub(r'\s+', ' ', full_text).strip()
        
        return full_text

    def extract_structured_nodes(self, latex_content: str, doc_id: str, source: str = None):
        """
        Extracts environments like theorem, definition, etc., by intelligently parsing the XML tree.
        """
        xml_output = latex_processor.run_latexml_on_content(latex_content)

        if not xml_output:
            print(f"WARNING: LaTeXML returned no content for doc_id: {doc_id}. Skipping.")
            return

        # Remove namespace to make parsing easier
        xml_output = re.sub(r' xmlns="[^"]+"', '', xml_output, count=1)

        try:
            root = ET.fromstring(xml_output)
        except ET.ParseError as e:
            print(f"FATAL: Could not parse XML for {doc_id}. Error: {e}")
            return

        # Find all theorem-like environments. LaTeXML groups them under the <theorem> tag.
        theorem_elements = root.findall(".//theorem")
        print(f"Found {len(theorem_elements)} theorem-like environments in {doc_id}")

        # Store nodes for later edge creation
        nodes_by_type = {}
        nodes_by_doc = {}
        
        # Also look for proof environments which might be tagged differently
        proof_elements = root.findall(".//proof")
        all_elements = theorem_elements + proof_elements
        
        print(f"Total environments found (including proofs): {len(all_elements)}")

        for element in all_elements:
            # Extract title if present
            title_element = element.find('title')
            title_text = ""
            
            if title_element is not None:
                # Get just the title text, avoiding duplicates
                title_parts = []
                if title_element.text:
                    title_parts.append(title_element.text.strip())
                for child in title_element:
                    if child.text:
                        title_parts.append(child.text.strip())
                    if child.tail:
                        title_parts.append(child.tail.strip())
                
                # Join and clean title
                title_text = " ".join(title_parts)
                title_text = re.sub(r'\s+', ' ', title_text).strip()
            
            # Determine the node type
            if element.tag == 'proof':
                node_type = 'proof'
                if not title_text:
                    title_text = "Proof"
            else:
                node_type = self._get_node_type_from_title(title_text)
            
            # Extract main content text, excluding the title to avoid duplication
            main_text = self._get_clean_text(element, exclude_title=True)

            if not main_text.strip():
                print(f"Skipping empty {node_type} in {doc_id}")
                continue

            # Generate a unique ID
            label = element.get('id', f"{node_type}-{uuid.uuid4().hex[:8]}")
            
            # Extract references
            refs = [ref.get('refid') for ref in element.findall('.//ref') if ref.get('refid')]
            
            # Generate embedding
            embedding = self.embedding_model.encode(main_text, convert_to_tensor=False)

            # Add node to graph
            self.graph.add_node(
                label,
                node_type=node_type,
                doc_id=doc_id,
                source=source,
                text=main_text,
                title=title_text,
                embedding=embedding
            )

            print(f"Added {node_type}: {title_text[:50]}... (text length: {len(main_text)})")

            # Store node in type and doc collections
            if node_type not in nodes_by_type:
                nodes_by_type[node_type] = []
            nodes_by_type[node_type].append(label)

            if doc_id not in nodes_by_doc:
                nodes_by_doc[doc_id] = []
            nodes_by_doc[doc_id].append(label)

            # Add reference edges
            for ref_label in refs:
                self.graph.add_edge(label, ref_label, edge_type='references')

        # Add edges between nodes of the same type
        for node_type, nodes in nodes_by_type.items():
            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    self.graph.add_edge(nodes[i], nodes[j], edge_type='same_type')

        # Add edges between nodes in the same document
        for doc_nodes in nodes_by_doc.values():
            for i in range(len(doc_nodes)):
                for j in range(i + 1, len(doc_nodes)):
                    self.graph.add_edge(doc_nodes[i], doc_nodes[j], edge_type='same_doc')

        if not self.graph.nodes:
            print(f"Skipping LaTeX file due to no structured content found: {source or doc_id}")

        print(f"Total nodes in graph after processing {doc_id}: {len(self.graph.nodes)}")
        print(f"Total edges in graph after processing {doc_id}: {len(self.graph.edges)}")

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
            # Use title if available, otherwise create from type
            concept_name = data.get('title', '')
            if not concept_name:
                concept_name = f"{data.get('node_type', 'unknown').title()} {node[-6:]}"
            
            blocks.append({
                'id': node,
                'type': data.get('node_type', 'unknown'),
                'text': data.get('text', ''),
                'doc_id': data.get('doc_id', ''),
                'source': data.get('source', ''),
                'embedding': data.get('embedding', None),
                'concept_name': concept_name
            })
        return blocks