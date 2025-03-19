import streamlit as st
import re
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import FancyArrowPatch
import pandas as pd
from IPython.display import HTML

# Set page configuration
st.set_page_config(
    page_title="Simplified Regex to Automata Converter",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
<style>
    .stButton button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 16px;
        border: none;
    }
    .stButton button:hover {
        background-color: #45a049;
    }
    .success-message {
        background-color: #4CAF50;
        color: white;
        padding: 10px;
        border-radius: 5px;
        text-align: center;
    }
    .error-message {
        background-color: #f44336;
        color: white;
        padding: 10px;
        border-radius: 5px;
        text-align: center;
    }
    .main-header {
        text-align: center;
        font-size: 36px;
        font-weight: bold;
        margin-bottom: 30px;
    }
    .sub-header {
        font-size: 24px;
        font-weight: bold;
        margin-top: 20px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">Simplified Regex to DFA, CFG & PDA Converter</h1>', unsafe_allow_html=True)

# Define regular expressions for better performance
regex_options = {
    "(a+b)*": "Any sequence of a's and b's (includes empty string)",
    "a*b*": "Any sequence of a's followed by any sequence of b's (includes empty string)"
}

# Thompson's algorithm for NFA construction - with ID-based states
class NFAState:
    _next_id = 0  # Class variable for generating unique IDs
    
    def __init__(self, is_final=False):
        self.transitions = {}  # Dictionary to store transitions {symbol: [states]}
        self.epsilon_transitions = []  # List to store epsilon transitions
        self.is_final = is_final
        self.state_id = NFAState._next_id  # Assign unique ID
        NFAState._next_id += 1

    def add_transition(self, symbol, state):
        if symbol in self.transitions:
            self.transitions[symbol].append(state)
        else:
            self.transitions[symbol] = [state]

    def add_epsilon_transition(self, state):
        self.epsilon_transitions.append(state)
    
    def __hash__(self):
        return hash(self.state_id)
    
    def __eq__(self, other):
        if isinstance(other, NFAState):
            return self.state_id == other.state_id
        return False

class NFA:
    def __init__(self, start_state, end_state):
        self.start_state = start_state
        self.end_state = end_state
        end_state.is_final = True
    
    @staticmethod
    def reset_state_ids():
        NFAState._next_id = 0

# Epsilon closure function - identifies all states reachable via epsilon transitions
def epsilon_closure(state, visited=None):
    if visited is None:
        visited = set()
    
    if state in visited:
        return visited
    
    visited.add(state)
    for eps_state in state.epsilon_transitions:
        epsilon_closure(eps_state, visited)
    
    return visited

# Get a set of state IDs for better serialization
def get_state_id_set(states):
    return frozenset(state.state_id for state in states)

# Convert regex to NFA
def regex_to_nfa(regex):
    """Convert a regular expression to an NFA using Thompson's construction algorithm."""
    
    # Reset state IDs to maintain consistency
    NFA.reset_state_ids()
    
    def handle_concatenation(nfa1, nfa2):
        nfa1.end_state.add_epsilon_transition(nfa2.start_state)
        nfa1.end_state.is_final = False
        return NFA(nfa1.start_state, nfa2.end_state)
    
    def handle_union(nfa1, nfa2):
        start = NFAState()
        end = NFAState(is_final=True)
        
        start.add_epsilon_transition(nfa1.start_state)
        start.add_epsilon_transition(nfa2.start_state)
        
        nfa1.end_state.add_epsilon_transition(end)
        nfa1.end_state.is_final = False
        
        nfa2.end_state.add_epsilon_transition(end)
        nfa2.end_state.is_final = False
        
        return NFA(start, end)
    
    def handle_kleene_star(nfa):
        start = NFAState()
        end = NFAState(is_final=True)
        
        start.add_epsilon_transition(nfa.start_state)
        start.add_epsilon_transition(end)
        
        nfa.end_state.add_epsilon_transition(nfa.start_state)
        nfa.end_state.add_epsilon_transition(end)
        nfa.end_state.is_final = False
        
        return NFA(start, end)
    
    def parse_regex(regex, i=0):
        nfas = []
        current_nfa = None
        
        while i < len(regex):
            c = regex[i]
            
            if c == '(':
                sub_nfa, i = parse_regex(regex, i + 1)
                if current_nfa is None:
                    current_nfa = sub_nfa
                else:
                    current_nfa = handle_concatenation(current_nfa, sub_nfa)
            
            elif c == ')':
                if current_nfa is not None:
                    return current_nfa, i
                else:
                    return NFA(NFAState(), NFAState(is_final=True)), i
            
            elif c == '+':
                if i + 1 < len(regex):
                    next_char = regex[i + 1]
                    if next_char == '(':
                        sub_nfa, i = parse_regex(regex, i + 2)
                        if current_nfa is None:
                            current_nfa = sub_nfa
                        else:
                            current_nfa = handle_union(current_nfa, sub_nfa)
                    else:
                        next_state = NFAState()
                        end_state = NFAState(is_final=True)
                        next_state.add_transition(next_char, end_state)
                        next_nfa = NFA(next_state, end_state)
                        if current_nfa is None:
                            current_nfa = next_nfa
                        else:
                            current_nfa = handle_union(current_nfa, next_nfa)
                        i += 1
            
            elif c == '*':
                if current_nfa is not None:
                    current_nfa = handle_kleene_star(current_nfa)
            
            else:  # Regular character
                state = NFAState()
                end_state = NFAState(is_final=True)
                state.add_transition(c, end_state)
                new_nfa = NFA(state, end_state)
                
                if current_nfa is None:
                    current_nfa = new_nfa
                else:
                    current_nfa = handle_concatenation(current_nfa, new_nfa)
            
            i += 1
        
        if current_nfa is None:
            start = NFAState()
            end = NFAState(is_final=True)
            start.add_epsilon_transition(end)
            current_nfa = NFA(start, end)
        
        return current_nfa, i
    
    # Remove whitespace and simplify regex
    regex = regex.replace(" ", "")
    
    # Handle simple cases
    if regex == "":
        start = NFAState()
        end = NFAState(is_final=True)
        start.add_epsilon_transition(end)
        return NFA(start, end)
    elif regex == "ε":
        start = NFAState()
        end = NFAState(is_final=True)
        start.add_epsilon_transition(end)
        return NFA(start, end)
    
    # Parse the regex
    nfa, _ = parse_regex(regex)
    return nfa

# Convert NFA to DFA using subset construction
def nfa_to_dfa(nfa):
    """Convert an NFA to a DFA using the subset construction algorithm."""
    
    alphabet = set()
    state_map = {}  # Map state IDs to NFAState objects
    
    # Collect all symbols and build state map
    queue = [nfa.start_state]
    visited = set()
    
    while queue:
        state = queue.pop(0)
        if state in visited:
            continue
            
        visited.add(state)
        state_map[state.state_id] = state
        
        for symbol in state.transitions:
            alphabet.add(symbol)
            for next_state in state.transitions[symbol]:
                if next_state not in visited:
                    queue.append(next_state)
        
        for eps_state in state.epsilon_transitions:
            if eps_state not in visited:
                queue.append(eps_state)
    
    # Start with epsilon closure of the start state
    start_states = epsilon_closure(nfa.start_state)
    start_ids = get_state_id_set(start_states)
    
    # Map NFA state ID sets to DFA states
    dfa_states = {start_ids: 0}  # Start state is 0
    dfa_transitions = {}
    dfa_final_states = set()
    
    # Check if the start state is also a final state
    if any(state.is_final for state in start_states):
        dfa_final_states.add(0)
    
    # Process queue with state IDs for better hashing
    queue = [start_ids]
    while queue:
        current_state_ids = queue.pop(0)
        current_dfa_state = dfa_states[current_state_ids]
        
        # Convert IDs back to NFAState objects for processing
        current_states = {state_map[state_id] for state_id in current_state_ids}
        
        for symbol in alphabet:
            next_states = set()
            
            # Get all states reachable by this symbol
            for state in current_states:
                if symbol in state.transitions:
                    next_states.update(state.transitions[symbol])
            
            # Add epsilon closures
            epsilon_states = set()
            for state in next_states:
                epsilon_states.update(epsilon_closure(state))
            
            if not epsilon_states:
                continue
                
            # Convert to IDs for hashing
            next_state_ids = get_state_id_set(epsilon_states)
            
            # Add new DFA state if needed
            if next_state_ids not in dfa_states:
                dfa_states[next_state_ids] = len(dfa_states)
                queue.append(next_state_ids)
                
                # Check if it's a final state
                if any(state_map[state_id].is_final for state_id in next_state_ids):
                    dfa_final_states.add(dfa_states[next_state_ids])
            
            # Add transition to DFA
            dfa_transitions[(current_dfa_state, symbol)] = dfa_states[next_state_ids]
    
    # Create the formal DFA structure (simple dictionary format for easy serialization)
    dfa = {
        'states': set(range(len(dfa_states))),
        'alphabet': alphabet,
        'transitions': dfa_transitions,
        'start_state': 0,
        'final_states': dfa_final_states
    }
    
    return dfa

# Convert DFA to CFG
def dfa_to_cfg(dfa):
    """Convert a DFA to a Context-Free Grammar"""
    cfg = {}
    
    # Start symbol corresponds to the start state
    start_symbol = f"S{dfa['start_state']}"
    
    # Create a production rule for each state
    for state in dfa['states']:
        # Non-terminal symbol for this state
        nt = f"S{state}"
        productions = []
        
        # Add transitions
        for symbol in sorted(dfa['alphabet']):
            if (state, symbol) in dfa['transitions']:
                next_state = dfa['transitions'][(state, symbol)]
                productions.append(f"{symbol}S{next_state}")
        
        # If it's a final state, also add an epsilon production
        if state in dfa['final_states']:
            productions.append("ε")
        
        # Add the production rule to the grammar
        cfg[nt] = productions
    
    return {
        'start_symbol': start_symbol,
        'productions': cfg
    }

# Convert DFA to PDA
def dfa_to_pda(dfa):
    """Convert a DFA to a Pushdown Automaton"""
    pda = {
        'states': dfa['states'],
        'input_alphabet': dfa['alphabet'],
        'stack_alphabet': {'Z0'} | {f"X{state}" for state in dfa['states']},
        'transitions': {},
        'start_state': dfa['start_state'],
        'start_stack_symbol': 'Z0',
        'final_states': dfa['final_states']
    }
    
    # Add transitions for each DFA transition
    for (state, symbol), next_state in sorted(dfa['transitions'].items()):
        # Push the next state onto the stack
        pda['transitions'][(state, symbol, 'Z0')] = (next_state, [f"X{next_state}", 'Z0'])
        
        # For each stack symbol (except Z0)
        for stack_state in sorted(dfa['states']):
            pda['transitions'][(state, symbol, f"X{stack_state}")] = (next_state, [f"X{next_state}", f"X{stack_state}"])
    
    # Add epsilon transitions for final states to pop the stack
    for final_state in sorted(dfa['final_states']):
        pda['transitions'][(final_state, 'ε', 'Z0')] = (final_state, ['Z0'])
        for stack_state in sorted(dfa['states']):
            pda['transitions'][(final_state, 'ε', f"X{stack_state}")] = (final_state, [])
    
    return pda

# Simulate a DFA on an input string
def simulate_dfa(dfa, input_string):
    """Simulate a DFA on an input string and return the states visited."""
    current_state = dfa['start_state']
    states_visited = [current_state]
    
    for symbol in input_string:
        if (current_state, symbol) in dfa['transitions']:
            current_state = dfa['transitions'][(current_state, symbol)]
            states_visited.append(current_state)
        else:
            return states_visited, False  # Invalid transition
    
    return states_visited, current_state in dfa['final_states']

# Visualize a DFA without caching
def visualize_dfa(dfa):
    """Visualize a DFA using networkx and matplotlib."""
    G = nx.DiGraph()
    
    # Add all states as nodes
    for state in dfa['states']:
        G.add_node(state, is_final=state in dfa['final_states'], is_start=state == dfa['start_state'])
    
    # Add transitions as edges
    for (state, symbol), next_state in sorted(dfa['transitions'].items()):
        # Check if an edge already exists
        if G.has_edge(state, next_state):
            # Update the label
            G[state][next_state]['label'] += f", {symbol}"
        else:
            G.add_edge(state, next_state, label=symbol)
    
    # Create a figure
    plt.figure(figsize=(10, 6))
    
    # Position the nodes
    pos = nx.spring_layout(G, seed=42)
    
    # Draw nodes
    node_colors = ['lightblue' if G.nodes[node]['is_final'] else 'white' for node in G.nodes]
    nx.draw_networkx_nodes(G, pos, node_size=700, node_color=node_colors, edgecolors='black')
    
    # Double circle for final states
    final_states = [state for state in dfa['states'] if state in dfa['final_states']]
    nx.draw_networkx_nodes(G, pos, nodelist=final_states, node_size=600, node_color='none', edgecolors='black')
    
    # Special marker for start state
    start_state = dfa['start_state']
    plt.annotate('', xy=pos[start_state], xytext=(pos[start_state][0]-0.1, pos[start_state][1]), 
                 arrowprops=dict(arrowstyle="->", color='black'))
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, width=1.5, arrowsize=20)
    
    # Add edge labels
    edge_labels = {(u, v): G[u][v]['label'] for u, v in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=12)
    
    # Add node labels (state numbers)
    node_labels = {node: f"q{node}" for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=12)
    
    plt.axis('off')
    
    return plt

# Create static frames for visualization instead of animation
def create_dfa_frames(dfa, input_string, states_visited):
    """Create a series of static frames of the DFA process instead of animation."""
    frames = []
    G = nx.DiGraph()
    
    # Add all states as nodes
    for state in dfa['states']:
        G.add_node(state, is_final=state in dfa['final_states'], is_start=state == dfa['start_state'])
    
    # Add transitions as edges
    for (state, symbol), next_state in dfa['transitions'].items():
        # Check if an edge already exists
        if G.has_edge(state, next_state):
            # Update the label
            G[state][next_state]['label'] += f", {symbol}"
            G[state][next_state]['symbols'] = G[state][next_state].get('symbols', []) + [symbol]
        else:
            G.add_edge(state, next_state, label=symbol, symbols=[symbol])
    
    # Position the nodes
    pos = nx.spring_layout(G, seed=42)
    
    # Generate each frame
    for i in range(len(states_visited) + 1):
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Determine highlight states and transitions
        highlight_states = [states_visited[min(i, len(states_visited)-1)]]
        
        highlight_transitions = []
        if i > 0 and i < len(states_visited):
            highlight_transitions = [(states_visited[i-1], states_visited[i])]
        
        # Draw nodes
        node_colors = ['lightgreen' if node in highlight_states else 
                       ('lightblue' if G.nodes[node]['is_final'] else 'white') 
                       for node in G.nodes]
        nx.draw_networkx_nodes(G, pos, node_size=700, node_color=node_colors, edgecolors='black', ax=ax)
        
        # Double circle for final states
        final_states = [state for state in dfa['states'] if state in dfa['final_states']]
        nx.draw_networkx_nodes(G, pos, nodelist=final_states, node_size=600, 
                              node_color='none', edgecolors='black', ax=ax)
        
        # Special marker for start state
        start_state = dfa['start_state']
        ax.annotate('', xy=pos[start_state], 
                   xytext=(pos[start_state][0]-0.1, pos[start_state][1]), 
                   arrowprops=dict(arrowstyle="->", color='black'))
        
        # Draw edges
        edges = G.edges()
        edge_colors = []
        
        for u, v in edges:
            if (u, v) in highlight_transitions:
                edge_colors.append('red')
            else:
                edge_colors.append('black')
        
        nx.draw_networkx_edges(G, pos, edgelist=edges, width=1.5, 
                              arrowsize=20, edge_color=edge_colors, ax=ax)
        
        # Add edge labels
        edge_labels = {(u, v): G[u][v]['label'] for u, v in G.edges()}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=12, ax=ax)
        
        # Add node labels (state numbers)
        node_labels = {node: f"q{node}" for node in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=12, ax=ax)
        
        # Display the current character being processed
        if i > 0 and i < len(input_string) + 1:
            char = input_string[i-1]
            current_state = states_visited[min(i-1, len(states_visited)-1)]
            next_state = states_visited[min(i, len(states_visited)-1)]
            
            # If there's a transition, show the symbol on the edge
            if current_state != next_state:
                ax.set_title(f"Processing: '{char}' | Transition: q{current_state} --{char}--> q{next_state}", 
                            fontsize=16)
            else:
                ax.set_title(f"Processing: '{char}' | No valid transition from q{current_state}", 
                            fontsize=16)
        elif i == 0:
            ax.set_title(f"Start: at state q{states_visited[0]}", fontsize=16)
        else:
            result = "Accepted" if states_visited[-1] in dfa['final_states'] else "Rejected"
            ax.set_title(f"Finished: String {result}", fontsize=16)
        
        ax.axis('off')
        frames.append(fig)
    
    return frames

# Create DFA from regex without caching
def get_dfa_from_regex(regex):
    """Create a DFA from a regex, avoiding caching issues."""
    # Reset NFA state IDs to ensure consistent generation
    NFA.reset_state_ids()
    
    # Convert regex to NFA
    nfa = regex_to_nfa(regex)
    
    # Convert NFA to DFA
    dfa = nfa_to_dfa(nfa)
    
    return dfa

# Display the regex conversion with step-by-step frames instead of animation
def display_regex_conversion(regex, input_string=None):
    """Display the converted regex as DFA, CFG, and PDA."""
    st.markdown(f'<h2 class="sub-header">Deterministic Finite Automaton for: {regex}</h2>', unsafe_allow_html=True)
    
    with st.spinner("Converting regex to automata..."):
        # Get a fresh DFA each time
        dfa = get_dfa_from_regex(regex)
        
        # Generate CFG and PDA
        cfg = dfa_to_cfg(dfa)
        pda = dfa_to_pda(dfa)
        
        # Create two containers for better layout
        static_container = st.container()
        animation_container = st.container()
        
        with static_container:
            st.markdown("### Static DFA Visualization")
            fig = visualize_dfa(dfa)
            st.pyplot(fig)
        
        # Visualize with input string if provided
        if input_string:
            states_visited, is_valid = simulate_dfa(dfa, input_string)
            st.write(f"Entered String: {input_string}")
            
            with animation_container:
                st.markdown("### Step-by-Step Animation")
                st.markdown("Use the slider below to see how the DFA processes the input string:")
                
                # Create frames with larger figure size
                frames = create_dfa_frames(dfa, input_string, states_visited)
                
                # Add a slider with more prominent styling
                frame_index = st.slider(
                    "Animation Step",
                    0,
                    len(frames)-1,
                    0,
                    key="animation_slider"
                )
                
                # Display the current frame with full width
                st.pyplot(frames[frame_index], use_container_width=True)
                
                # Show current step information
                if frame_index == 0:
                    st.info("Initial state")
                elif frame_index < len(input_string) + 1:
                    st.info(f"Processing character: {input_string[frame_index-1]}")
                else:
                    st.info("Final state")
                
                # Show validation result
                if is_valid:
                    st.markdown(f'<div class="success-message">The string \'{input_string}\' is valid for the DFA.</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="error-message">The string \'{input_string}\' is NOT valid for the DFA.</div>', unsafe_allow_html=True)
        
        # Show CFG and PDA in expanders below the animation
        with st.expander("Context-Free Grammar (CFG) Representation"):
            st.markdown("### Productions")
            st.write(f"Start Symbol: {cfg['start_symbol']}")
            for nt, productions in sorted(cfg['productions'].items()):
                production_str = " | ".join(productions) if productions else "ε"
                st.write(f"{nt} → {production_str}")
        
        with st.expander("Pushdown Automaton (PDA) Representation"):
            st.markdown("### States")
            st.write(f"States: {', '.join([f'q{s}' for s in sorted(pda['states'])])}")
            st.write(f"Start State: q{pda['start_state']}")
            st.write(f"Final States: {', '.join([f'q{s}' for s in sorted(pda['final_states'])])}")
            
            st.markdown("### Alphabets")
            st.write(f"Input Alphabet: {', '.join(sorted(pda['input_alphabet']))}")
            st.write(f"Stack Alphabet: {', '.join(sorted(list(pda['stack_alphabet'])[:5]))}...")

# Initialize session state for input string
if 'input_string' not in st.session_state:
    st.session_state.input_string = None

# Main app logic
st.markdown('<h2 class="sub-header">Select a Regular Expression</h2>', unsafe_allow_html=True)
selected_regex = st.selectbox("", list(regex_options.keys()), format_func=lambda x: regex_options[x])

# Pre-computed examples for quick access
examples = {
    "(a+b)*": ["ab", "aabb", "baba", ""],
    "a*b*": ["aaabbb", "aaa", "bbb", ""]
}

# Display the conversion for the selected regex
display_regex_conversion(selected_regex)

# Section for string validation
st.markdown('<h2 class="sub-header">Enter a string to check its validity for displayed DFA</h2>', unsafe_allow_html=True)

col1, col2 = st.columns([3, 1])

with col1:
    input_string = st.text_input("", placeholder="Enter a string (e.g., abababba)", key="input_string_field")
    
    # Show example strings for the selected regex
    st.markdown("**Examples to try:**")
    example_buttons = st.columns(len(examples[selected_regex]))
    
    for i, example in enumerate(examples[selected_regex]):
        # Ensure each button has a unique key
        display_text = example if example != "" else "empty string"
        if example_buttons[i].button(display_text, key=f"example_{selected_regex}_{i}"):
            st.session_state.input_string = example
            st.rerun()  # Use rerun instead of experimental_rerun

with col2:
    if st.button("Validate", use_container_width=True, key="validate_button"):
        if input_string:
            # Display the conversion with frames for the input string
            display_regex_conversion(selected_regex, input_string)
        else:
            st.error("Please enter a string to validate.")

# If there's an example input string set from button clicks
if st.session_state.input_string:
    display_regex_conversion(selected_regex, st.session_state.input_string)
    # Clear the session state after use
    st.session_state.input_string = None

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>Simplified Regex to Automata Converter | Created with Streamlit</p>
</div>
""", unsafe_allow_html=True)