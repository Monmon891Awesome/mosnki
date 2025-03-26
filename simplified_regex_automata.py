import streamlit as st
import re
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import FancyArrowPatch
import pandas as pd
import base64
from io import BytesIO
import numpy as np
import time
from PIL import Image


# ======== CONFIGURATION AND PAGE SETUP ========
st.set_page_config(
    page_title="Regex to Automata Converter",
    page_icon="🔄",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ======== CACHE DECORATORS FOR PERFORMANCE ========
@st.cache_data(ttl=3600)
def get_css():
    """Load CSS from a file or string"""
    return """
    /* Modern design system with consistent color palette */
    :root {
        --primary: #4361ee;
        --primary-hover: #3a56d4;
        --secondary: #4cc9f0;
        --success: #06d6a0;
        --warning: #ffd166;
        --danger: #ef476f;
        --light: #f8f9fa;
        --dark: #212529;
        --gray: #6c757d;
        --background: #f7f7fc;
        --card: #ffffff;
        --transition: all 0.3s ease;
        --shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        --shadow-lg: 0 10px 15px rgba(0, 0, 0, 0.1);
        --border-radius: 8px;
        --font-main: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Additional CSS content from the CSS artifact */
    """

@st.cache_data(ttl=3600)
def get_regex_options():
    """Return predefined regex options with descriptions"""
    return {
        "(a+b)*": "Any sequence of a's and b's (includes empty string)",
        "a*b*": "Any sequence of a's followed by any sequence of b's (includes empty string)",
        "(a+b)*abb(a+b)*": "Any string containing 'abb' as a substring",
        "a*ba*ba*": "Strings with exactly two b's",
        "(aa+bb)*((ab+ba)(aa+bb)*(ab+ba)(aa+bb)*)*": "Strings with an even number of a's and b's",
        "a+b": "Either a single 'a' or a single 'b'",
        "a(ba)*": "Strings like 'a', 'aba', 'ababa', etc."
    }

@st.cache_data(ttl=600)
def get_examples():
    """Return example strings for each regex"""
    return {
        "(a+b)*": ["ab", "aabb", "baba", ""],
        "a*b*": ["aaabbb", "aaa", "bbb", ""],
        "(a+b)*abb(a+b)*": ["abb", "aabba", "bbabb", "abbb"],
        "a*ba*ba*": ["bb", "abb", "bab", "abba"],
        "(aa+bb)*((ab+ba)(aa+bb)*(ab+ba)(aa+bb)*)*": ["aabb", "abba", "aaabbb", "aababb"],
        "a+b": ["a", "b"],
        "a(ba)*": ["a", "aba", "ababa", "abababa"]
    }

# ======== NFA CONSTRUCTION CLASSES ========
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

# ======== ALGORITHM FUNCTIONS ========
@st.cache_data(ttl=600)
def epsilon_closure(state_id, epsilon_transitions_map):
    """Optimized epsilon closure function using state IDs"""
    result = {state_id}
    stack = [state_id]
    
    while stack:
        current = stack.pop()
        for next_state in epsilon_transitions_map.get(current, []):
            if next_state not in result:
                result.add(next_state)
                stack.append(next_state)
    
    return frozenset(result)

@st.cache_data(ttl=600)
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
    
    # Normalize regex
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

@st.cache_data(ttl=600)
def nfa_to_dfa(_nfa):
    """Optimized NFA to DFA conversion using subset construction algorithm."""
    
    # Create efficient maps for transitions
    symbol_map = {}
    epsilon_map = {}
    state_map = {}
    alphabet = set()
    
    # Collect states and transitions
    queue = [_nfa.start_state]
    visited = set()
    
    while queue:
        state = queue.pop(0)
        state_id = state.state_id
        
        if state_id in visited:
            continue
        
        visited.add(state_id)
        state_map[state_id] = state
        
        # Regular transitions
        for symbol, next_states in state.transitions.items():
            alphabet.add(symbol)
            
            if state_id not in symbol_map:
                symbol_map[state_id] = {}
            
            if symbol not in symbol_map[state_id]:
                symbol_map[state_id][symbol] = []
            
            for next_state in next_states:
                symbol_map[state_id][symbol].append(next_state.state_id)
                if next_state.state_id not in visited:
                    queue.append(next_state)
        
        # Epsilon transitions
        if state.epsilon_transitions:
            epsilon_map[state_id] = [s.state_id for s in state.epsilon_transitions]
            for eps_state in state.epsilon_transitions:
                if eps_state.state_id not in visited:
                    queue.append(eps_state)
    
    # Start with epsilon closure of the start state
    start_closure = epsilon_closure(_nfa.start_state.state_id, epsilon_map)
    
    # Map NFA state ID sets to DFA states
    dfa_states = {start_closure: 0}  # Start state is 0
    dfa_transitions = {}
    dfa_final_states = set()
    
    # Check if the start state is also a final state
    if any(state_map[state_id].is_final for state_id in start_closure):
        dfa_final_states.add(0)
    
    # Process queue with state IDs for better performance
    queue = [start_closure]
    while queue:
        current_state_ids = queue.pop(0)
        current_dfa_state = dfa_states[current_state_ids]
        
        # Process each symbol in the alphabet
        for symbol in alphabet:
            next_states = set()
            
            # Get all states reachable by this symbol
            for state_id in current_state_ids:
                if state_id in symbol_map and symbol in symbol_map[state_id]:
                    next_states.update(symbol_map[state_id][symbol])
            
            # Apply epsilon closures to all next states
            epsilon_states = set()
            for state_id in next_states:
                epsilon_states.update(epsilon_closure(state_id, epsilon_map))
            
            if not epsilon_states:
                continue
            
            # Convert to frozenset for hashing
            next_state_ids = frozenset(epsilon_states)
            
            # Add new DFA state if needed
            if next_state_ids not in dfa_states:
                dfa_states[next_state_ids] = len(dfa_states)
                queue.append(next_state_ids)
                
                # Check if it's a final state
                if any(state_map[state_id].is_final for state_id in next_state_ids):
                    dfa_final_states.add(dfa_states[next_state_ids])
            
            # Add transition to DFA
            dfa_transitions[(current_dfa_state, symbol)] = dfa_states[next_state_ids]
    
    # Create the formal DFA structure
    dfa = {
        'states': set(range(len(dfa_states))),
        'alphabet': alphabet,
        'transitions': dfa_transitions,
        'start_state': 0,
        'final_states': dfa_final_states
    }
    
    return dfa

@st.cache_data(ttl=600)
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

@st.cache_data(ttl=600)
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

@st.cache_data(ttl=600)
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

@st.cache_data(ttl=600)
def get_dfa_from_regex(regex):
    """Create a DFA from a regex, avoiding caching issues."""
    # Reset NFA state IDs to ensure consistent generation
    NFA.reset_state_ids()
    
    # Convert regex to NFA
    nfa = regex_to_nfa(regex)
    
    # Convert NFA to DFA
    dfa = nfa_to_dfa(nfa)
    
    return dfa

@st.cache_data(ttl=600)
def get_all_automata_from_regex(regex):
    """Create DFA, CFG, and PDA from a regex."""
    dfa = get_dfa_from_regex(regex)
    cfg = dfa_to_cfg(dfa)
    pda = dfa_to_pda(dfa)
    
    return dfa, cfg, pda

# ======== VISUALIZATION FUNCTIONS ========
@st.cache_data(ttl=600)
def visualize_dfa(dfa, highlight_states=None, highlight_transitions=None, title=None):
    """Visualize a DFA using networkx and matplotlib."""
    # Create a directed graph
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
    
    # Set up figure
    plt.figure(figsize=(10, 7))
    
    # Use a deterministic layout algorithm
    pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)
    
    # Determine node colors based on state types and highlights
    highlight_states = highlight_states or []
    highlight_transitions = highlight_transitions or []
    
    node_colors = []
    for node in G.nodes():
        if node in highlight_states:
            node_colors.append('#4361ee')  # Primary color for highlighted states
        elif G.nodes[node]['is_final']:
            node_colors.append('#06d6a0')  # Success color for final states
        else:
            node_colors.append('#ffffff')  # White for normal states
    
    # Draw nodes with custom styling
    nx.draw_networkx_nodes(G, pos, node_size=800, node_color=node_colors, 
                          edgecolors='black', linewidths=2, alpha=0.9)
    
    # Draw double circle for final states
    final_states = [state for state in dfa['states'] if state in dfa['final_states']]
    nx.draw_networkx_nodes(G, pos, nodelist=final_states, node_size=700, 
                          node_color='none', edgecolors='black', linewidths=2)
    
    # Determine edge colors
    edge_colors = []
    edge_widths = []
    
    for u, v in G.edges():
        if (u, v) in highlight_transitions:
            edge_colors.append('#ef476f')  # Highlighted edge
            edge_widths.append(3)
        else:
            edge_colors.append('#212529')  # Default edge color
            edge_widths.append(1.5)
    
    # Draw edges with styled arrows
    nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color=edge_colors,
                          arrowstyle='->', arrowsize=20, connectionstyle="arc3,rad=0.1")
    
    # Add special marker for start state
    start_state = dfa['start_state']
    plt.annotate('', xy=pos[start_state], xytext=(pos[start_state][0]-0.15, pos[start_state][1]), 
                arrowprops=dict(arrowstyle="->", color='black', lw=2))
    
    # Add edge labels
    edge_labels = {(u, v): G[u][v]['label'] for u, v in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=12, 
                                font_family='sans-serif', font_weight='bold')
    
    # Add node labels
    node_labels = {node: f"q{node}" for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=14, 
                           font_family='sans-serif', font_weight='bold')
    
    plt.axis('off')
    
    # Add title if provided
    if title:
        plt.title(title, fontsize=16, fontweight='bold', pad=20, fontfamily='sans-serif')
    
    # Adjust margins
    plt.tight_layout()
    
    return plt

def create_dfa_animation_frames(dfa, input_string, max_frames=20):
    """Create static frames for a DFA simulation animation."""
    states_visited, is_accepted = simulate_dfa(dfa, input_string)
    
    # Create a series of frames showing the DFA processing
    frames = []
    
    # Add initial state frame
    highlight_states = [states_visited[0]]
    highlight_transitions = []
    title = "Initial State"
    fig = visualize_dfa(dfa, highlight_states, highlight_transitions, title)
    frames.append(fig)
    
    # Add frames for each transition
    for i in range(1, min(len(states_visited), max_frames)):
        highlight_states = [states_visited[i]]
        highlight_transitions = [(states_visited[i-1], states_visited[i])]
        
        if i < len(input_string):
            title = f"Processing: '{input_string[i-1]}'"
        else:
            title = "Final State"
        
        fig = visualize_dfa(dfa, highlight_states, highlight_transitions, title)
        frames.append(fig)
    
    # Add final state frame if not already included
    if len(states_visited) > max_frames:
        highlight_states = [states_visited[-1]]
        highlight_transitions = []
        result = "Accepted" if is_accepted else "Rejected"
        title = f"Final State - String {result}"
        fig = visualize_dfa(dfa, highlight_states, highlight_transitions, title)
        frames.append(fig)
    
    return frames, is_accepted

def fig_to_base64(fig):
    """Convert a matplotlib figure to base64 encoded string."""
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return img_str

# ======== UI COMPONENTS ========
def render_header():
    """Render the app header with logo and title."""
    st.markdown(
        '<div class="header-container">'
        '<h1 class="main-header">✨ Regex to Automata Converter</h1>'
        '<p class="header-subtitle">Transform regular expressions into DFA, CFG, and PDA</p>'
        '</div>',
        unsafe_allow_html=True
    )

def render_regex_selector():
    """Render the regex selection interface with custom regex input option."""
    st.markdown('<h2 class="sub-header">Select or Create a Regular Expression</h2>', unsafe_allow_html=True)
    
    # Create card effect with columns
    col1, col2 = st.columns([2, 3])
    
    with col1:
        # Option to use predefined regex or custom one
        regex_type = st.radio(
            "Choose regex source:",
            options=["Predefined Examples", "Custom Regex"],
            horizontal=True
        )
    
    # Setup container for regex selection
    regex_container = st.container()
    
    with regex_container:
        if regex_type == "Predefined Examples":
            # Use predefined regex options
            selected_regex = st.selectbox(
                "Select a regular expression:",
                options=list(get_regex_options().keys()),
                format_func=lambda x: f"{x} - {get_regex_options()[x]}"
            )
        else:
            # Input custom regex
            selected_regex = st.text_input(
                "Enter your custom regular expression:",
                placeholder="Example: (a+b)*abb",
                help="Use '+' for union, '*' for Kleene star, and parentheses for grouping"
            )
            
            if not selected_regex:
                st.info("Please enter a valid regular expression or choose a predefined one.")
    
    # Display regex explanation if a predefined one is selected
    if regex_type == "Predefined Examples" and selected_regex:
        st.markdown(
            f'<div class="regex-explanation">'
            f'<strong>Description:</strong> {get_regex_options()[selected_regex]}'
            f'</div>',
            unsafe_allow_html=True
        )
    
    return selected_regex

def render_string_input_section(regex):
    """Render the section for string input and validation."""
    st.markdown('<h2 class="sub-header">Test String Validation</h2>', unsafe_allow_html=True)
    
    # Create columns for input and examples
    col1, col2 = st.columns([3, 2])
    
    with col1:
        input_string = st.text_input(
            "Enter a string to validate:",
            placeholder="Example: abba",
            help="The string will be checked against the automaton"
        )
        
        validate_button = st.button("Validate String", type="primary", use_container_width=True)
    
    with col2:
        st.markdown("**Quick Test Examples:**")
        
        # Get examples for the selected regex
        examples = get_examples().get(regex, ["ab", ""])
        
        # Create a grid of example buttons
        example_cols = st.columns(len(examples))
        
        # Track if any example was clicked
        example_clicked = False
        selected_example = None
        
        for i, example in enumerate(examples):
            display_text = example if example != "" else "ε (empty string)"
            if example_cols[i].button(display_text, key=f"example_{regex}_{i}"):
                example_clicked = True
                selected_example = example
    
    # Return input string (either from text input or example button)
    if example_clicked and selected_example is not None:
        return selected_example, True
    else:
        return input_string, validate_button

def render_automaton_visualization(regex, input_string=None, validate=False):
    """Render the automaton visualization with tabs for DFA, CFG, and PDA."""
    if not regex:
        st.warning("Please select or enter a regular expression.")
        return
    
    # Show a loading spinner during computation
    with st.spinner("Computing automata..."):
        try:
            # Get all automata from the regex
            dfa, cfg, pda = get_all_automata_from_regex(regex)
            
            # Create tab layout for different representations
            tab1, tab2, tab3 = st.tabs(["DFA Visualization", "Context-Free Grammar", "Pushdown Automaton"])
            
            with tab1:
                st.markdown(f"### Deterministic Finite Automaton for: `{regex}`")
                
                # If we have an input string and should validate it
                if input_string is not None and validate:
                    # Simulate the DFA
                    states_visited, is_accepted = simulate_dfa(dfa, input_string)
                    
                    # Create frames for animation
                    frames, _ = create_dfa_animation_frames(dfa, input_string)
                    
                    # Convert frames to base64 for display
                    frame_data = [fig_to_base64(fig) for fig in frames]
                    
                    # Create manual slideshow using columns
                    st.markdown("#### Step-by-Step DFA Processing")
                    
                    # Display string and result
                    if is_accepted:
                        st.success(f"The string '{input_string if input_string else 'ε (empty string)'}' is ACCEPTED by the DFA.")
                    else:
                        st.error(f"The string '{input_string if input_string else 'ε (empty string)'}' is REJECTED by the DFA.")
                    
                    # Create a slider for frame navigation
                    frame_index = st.slider("Processing Step", 0, len(frame_data) - 1, 0)
                    
                    # Display the current frame
                    st.markdown(
                        f'<div class="frame-container"><img src="data:image/png;base64,{frame_data[frame_index]}" width="100%"></div>',
                        unsafe_allow_html=True
                    )
                    
                    # Add controls for animation
                    cols = st.columns([1, 1, 1])
                    
                    # Play button logic would go here in a real app
                    # This is simplified for the example
                    cols[1].markdown(
                        '<div style="text-align:center">'
                        '<span style="font-size:0.8rem">Use the slider to navigate through steps</span>'
                        '</div>',
                        unsafe_allow_html=True
                    )
                else:
                    # Just show the static DFA
                    fig = visualize_dfa(dfa)
                    st.pyplot(fig)
            
            with tab2:
                st.markdown(f"### Context-Free Grammar for: `{regex}`")
                
                # Display CFG details
                st.markdown("#### Production Rules")
                st.info(f"Start Symbol: {cfg['start_symbol']}")
                
                # Create a table for productions
                productions_data = []
                for nt, productions in sorted(cfg['productions'].items()):
                    production_str = " | ".join(productions) if productions else "ε"
                    productions_data.append([nt, production_str])
                
                # Display as DataFrame for nice formatting
                df = pd.DataFrame(productions_data, columns=["Non-terminal", "Productions"])
                st.table(df)
                
                # Offer explanation
                with st.expander("How to read the CFG"):
                    st.markdown("""
                    A Context-Free Grammar consists of:
                    - **Non-terminal symbols**: These are represented as S0, S1, etc.
                    - **Terminal symbols**: These are the input alphabet (a, b, etc.)
                    - **Production rules**: These show how non-terminals can be replaced
                    - **Start symbol**: This is where the derivation begins
                    
                    Each rule shows how a non-terminal (left side) can be replaced with a sequence 
                    of terminals and non-terminals (right side). The | symbol separates alternative productions.
                    """)
            
            with tab3:
                st.markdown(f"### Pushdown Automaton for: `{regex}`")
                
                # Display PDA details in a more organized way
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### States and Alphabets")
                    st.info(f"Start State: q{pda['start_state']}")
                    st.info(f"Final States: {', '.join([f'q{s}' for s in sorted(pda['final_states'])])}")
                
                with col2:
                    st.markdown("#### Alphabets")
                    st.info(f"Input Alphabet: {', '.join(sorted(pda['input_alphabet']))}")
                    st.info(f"Initial Stack Symbol: {pda['start_stack_symbol']}")
                
                # Show some transitions (limited to avoid overwhelming)
                st.markdown("#### Sample Transitions")
                
                transitions_data = []
                for i, ((state, symbol, stack), (next_state, new_stack)) in enumerate(sorted(pda['transitions'].items())):
                    if i >= 10:  # Limit to first 10 transitions
                        break
                    
                    stack_str = stack
                    new_stack_str = ", ".join(new_stack) if new_stack else "ε (pop)"
                    
                    transitions_data.append([
                        f"q{state}", 
                        symbol if symbol != 'ε' else "ε (epsilon)", 
                        stack_str,
                        f"q{next_state}",
                        new_stack_str
                    ])
                
                # Display as DataFrame
                df = pd.DataFrame(
                    transitions_data, 
                    columns=["Current State", "Input Symbol", "Stack Top", "Next State", "New Stack"]
                )
                st.table(df)
                
                # Add note about transition count
                if len(pda['transitions']) > 10:
                    st.info(f"Showing 10 of {len(pda['transitions'])} transitions.")
                
                # Offer explanation
                with st.expander("How to read the PDA"):
                    st.markdown("""
                    A Pushdown Automaton (PDA) consists of:
                    - **States**: These are represented as q0, q1, etc.
                    - **Input alphabet**: The symbols that can be read from the input
                    - **Stack alphabet**: The symbols that can be pushed/popped from the stack
                    - **Transitions**: Based on current state, input symbol, and stack top
                    - **Start state and stack symbol**: Where processing begins
                    - **Final states**: Accepting states
                    
                    Each transition shows:
                    1. Current state
                    2. Input symbol to read (or ε for no input)
                    3. Symbol to pop from stack
                    4. Next state to move to
                    5. Symbols to push onto stack (rightmost pushed first)
                    """)
        
        except Exception as e:
            st.error(f"Error processing the regular expression: {str(e)}")
            st.info("Please check your regex syntax and try again.")

def render_footer():
    """Render the app footer with additional information."""
    st.markdown("---")
    st.markdown(
        '<div class="footer">'
        '<p>Regex to Automata Converter | Enhanced with Modern UI/UX</p>'
        '<p><small>📖 Uses Thompson\'s Construction Algorithm for NFA</small></p>'
        '</div>',
        unsafe_allow_html=True
    )

# ======== MAIN APP FUNCTION ========
def main():
    """Main function to run the Streamlit app."""
    # Apply custom CSS
    st.markdown(f'<style>{get_css()}</style>', unsafe_allow_html=True)
    
    # Render header
    render_header()
    
    # Sidebar with additional info and settings
    with st.sidebar:
        st.markdown("### About")
        st.markdown("""
        This application converts regular expressions into:
        - Deterministic Finite Automaton (DFA)
        - Context-Free Grammar (CFG)
        - Pushdown Automaton (PDA)
        
        You can also validate strings against the generated automaton.
        """)
        
        st.markdown("### Syntax Guide")
        st.markdown("""
        - `a`, `b`, etc.: Basic symbols
        - `+`: Union (OR) operation
        - `*`: Kleene star (0 or more)
        - `()`: Grouping
        - `ε`: Empty string (represented as "")
        """)
        
        # Optional settings
        st.markdown("### Settings")
        animation_speed = st.slider("Animation Speed", 0.5, 3.0, 1.0, 0.1)
        dark_mode = st.toggle("Dark Mode", value=False)
        
        # Apply dark mode if selected
        if dark_mode:
            st.markdown("""
            <style>
            :root {
                --background: #1e1e2e;
                --card: #2a2a3c;
                --dark: #e0e0e0;
                --light: #2a2a3c;
                --gray: #a0a0a0;
            }
            </style>
            """, unsafe_allow_html=True)
    
    # Main content area
    container = st.container()
    
    with container:
        # Get selected regex
        selected_regex = render_regex_selector()
        
        # Only proceed if we have a regex
        if selected_regex:
            # Get input string and validation flag
            input_string, validate = render_string_input_section(selected_regex)
            
            # Render the automaton visualization
            render_automaton_visualization(selected_regex, input_string, validate)
    
    # Render footer
    render_footer()

# ======== APP ENTRY POINT ========
if __name__ == "__main__":
    main()