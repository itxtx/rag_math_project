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

    def _get_node_type_from_element(self, element) -> str:
        """
        Determines the actual environment type from the element.
        LaTeXML converts all theorem-like environments to <theorem> tags,
        but we can infer the actual type from the title or attributes.
        """
        # First check if there's a 'class' attribute that might indicate the type
        elem_class = element.get('class', '')
        if elem_class:
            for env_type in ['definition', 'lemma', 'proposition', 'corollary', 'example', 'remark']:
                if env_type in elem_class.lower():
                    return env_type
        
        # Check the title element
        title_elem = element.find('title')
        if title_elem is not None:
            title_text = self._get_clean_text(title_elem, exclude_title=False).lower()
            
            # Look for environment indicators in the title
            for env_type in ['definition', 'lemma', 'proposition', 'corollary', 'example', 'remark', 'theorem']:
                if title_text.startswith(env_type):
                    return env_type
            
            # Check for common patterns like "Def.", "Prop.", etc.
            if title_text.startswith(('def.', 'defn.')):
                return 'definition'
            elif title_text.startswith('prop.'):
                return 'proposition'
            elif title_text.startswith('cor.'):
                return 'corollary'
            elif title_text.startswith('lem.'):
                return 'lemma'
            elif title_text.startswith('ex.'):
                return 'example'
            elif title_text.startswith('rem.'):
                return 'remark'
            elif title_text.startswith('thm.'):
                return 'theorem'
        
        # Check the content for clues
        content = self._get_clean_text(element, exclude_title=True).lower()
        if content.startswith('we define') or content.startswith('define') or 'is defined as' in content[:100]:
            return 'definition'
        
        # Default to theorem if we can't determine
        return 'theorem'

    def _extract_math_latex(self, math_element) -> str:
        """Extract LaTeX from a Math element."""
        tex = math_element.get('tex')
        if tex:
            # Determine if it's inline or display math
            mode = math_element.get('mode', 'inline')
            if mode == 'inline':
                return f"${tex}$"
            else:
                return f"$${tex}$$"
        return ""

    def _clean_extracted_text(self, text: str) -> str:
        """Clean up extracted text by removing artifacts and fixing spacing."""
        # Remove item numbers and tags like "1. 1 item 1"
        text = re.sub(r'\b\d+\.\s*\d+\s*item\s*\d+\s*', '', text)
        
        # Remove standalone numbers that appear to be labels
        text = re.sub(r'(?<![.$])\b\d+\b(?![.$])', '', text)
        
        # Clean up multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing spaces
        text = text.strip()
        
        return text

    def _get_clean_text(self, element, exclude_title=True) -> str:
        """
        Extracts text from an XML element, properly handling LaTeX math elements.
        """
        if element is None:
            return ""
        
        text_parts = []
        
        def process_element(elem, skip_title=False):
            """Process an element and its children to extract text."""
            # Skip title elements if requested
            if skip_title and elem.tag == 'title':
                return
            
            # Skip tags that are just formatting hints
            if elem.tag in ['tag', 'tags']:
                return
                
            # Add text before the element
            if elem.text and elem.text.strip():
                text_parts.append(elem.text.strip())
            
            # Process child elements
            for child in elem:
                if exclude_title and child.tag == 'title':
                    continue
                    
                if child.tag == 'Math':
                    # Extract the full LaTeX expression
                    math_text = self._extract_math_latex(child)
                    if math_text:
                        text_parts.append(math_text)
                elif child.tag in ['tag', 'tags']:
                    # Skip these formatting elements
                    continue
                elif child.tag == 'ERROR':
                    # Handle LaTeX commands that couldn't be processed
                    if child.text and child.text.strip():
                        text_parts.append(child.text.strip())
                else:
                    # Recursively process other elements
                    process_element(child, skip_title=False)
                
                # Add text after the element (tail)
                if child.tail and child.tail.strip():
                    text_parts.append(child.tail.strip())
        
        # Process the root element
        process_element(element, skip_title=exclude_title)
        
        # Join all parts with spaces
        full_text = ' '.join(text_parts)
        
        # Clean up the extracted text
        full_text = self._clean_extracted_text(full_text)
        
        return full_text

    def extract_structured_nodes(self, latex_content: str, doc_id: str, source: str = None):
        """
        Extracts environments like theorem, definition, etc., by intelligently parsing the XML tree.
        """
        try:
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

            # Find all theorem-like environments
            # LaTeXML converts most theorem-like environments to <theorem>
            theorem_elements = root.findall(".//theorem")
            proof_elements = root.findall(".//proof")
            
            # Also look for other possible tags LaTeXML might use
            para_elements = root.findall(".//para[@class]")  # Sometimes definitions are in para with class
            
            all_elements = theorem_elements + proof_elements
            
            # Filter para elements to find theorem-like content
            for para in para_elements:
                para_class = para.get('class', '')
                if any(env in para_class for env in ['definition', 'lemma', 'example', 'remark']):
                    all_elements.append(para)
            
            print(f"Found {len(all_elements)} structured environments in {doc_id}")
            
            # Store nodes for later edge creation
            nodes_by_type = {}
            nodes_by_doc = {}
            
            # Count environments by actual type
            env_counts = {}

            for element in all_elements:
                try:
                    # Determine the actual node type
                    if element.tag == 'proof':
                        node_type = 'proof'
                    else:
                        node_type = self._get_node_type_from_element(element)
                    
                    # Count this environment type
                    env_counts[node_type] = env_counts.get(node_type, 0) + 1
                    
                    # Extract title if present
                    title_element = element.find('title')
                    title_text = ""
                    
                    if title_element is not None:
                        title_text = self._get_clean_text(title_element, exclude_title=False)
                        # Remove the environment type prefix from title
                        for prefix in ['Definition', 'Theorem', 'Lemma', 'Proposition', 'Corollary', 'Example', 'Remark', 'Proof']:
                            if title_text.startswith(prefix):
                                # Keep the number and parenthetical content if present
                                title_text = title_text[len(prefix):].strip()
                                if title_text.startswith('.'):
                                    title_text = title_text[1:].strip()
                                break
                        
                        if not title_text:
                            title_text = node_type.capitalize()
                    else:
                        title_text = node_type.capitalize()
                    
                    # Extract main content text, excluding the title to avoid duplication
                    main_text = self._get_clean_text(element, exclude_title=True)

                    if not main_text.strip():
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
                        
                except Exception as e:
                    print(f"Error processing element in {doc_id}: {e}")
                    continue

            # Print environment type summary
            print(f"\nEnvironment type breakdown for {doc_id}:")
            for env_type, count in sorted(env_counts.items()):
                print(f"  {env_type.capitalize()}s: {count}")

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
            
        except Exception as e:
            print(f"Error in extract_structured_nodes for {doc_id}: {e}")
            import traceback
            traceback.print_exc()

    def save_graph_and_embeddings(self, graph_path, embeddings_path):
        """Saves the final graph and initial embeddings."""
        print(f"Saving knowledge graph to {graph_path}")
        
        # Create directories if they don't exist
        os.makedirs(os.path.dirname(graph_path), exist_ok=True)
        os.makedirs(os.path.dirname(embeddings_path), exist_ok=True)
        
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
            if concept_name:
                # Prepend the environment type to make it clearer
                concept_name = f"{data.get('node_type', 'unknown').capitalize()} {concept_name}"
            else:
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