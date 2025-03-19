import streamlit as st
import re
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import FancyArrowPatch
import time
import pandas as pd
from IPython.display import HTML
import base64

# Set page configuration
st.set_page_config(
    page_title="Regex to Automata Converter",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS - Enhanced for better user experience
st.markdown("""
<style>
    .stButton button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 16px;
        border: none;
        transition: background-color 0.3s;
    }
    .stButton button:hover {
        background-color: #45a049;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }
    .expander-header {
        font-size: 20px;
        font-weight: bold;
    }
    .success-message {
        background-color: #4CAF50;
        color: white;
        padding: 15px;
        border-radius: 5px;
        text-align: center;
        margin: 15px 0;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .error-message {
        background-color: #f44336;
        color: white;
        padding: 15px;
        border-radius: 5px;
        text-align: center;
        margin: 15px 0;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .main-header {
        text-align: center;
        font-size: 40px;
        font-weight: bold;
        margin-bottom: 30px;
        color: #2E7D32;
        text-shadow: 1px 1px 2px #CCCCCC;
    }
    .sub-header {
        font-size: 24px;
        font-weight: bold;
        margin-top: 25px;
        margin-bottom: 15px;
        color: #1B5E20;
        border-bottom: 2px solid #E8F5E9;
        padding-bottom: 8px;
    }
    .stExpander {
        border: 1px solid #E8F5E9;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    .state-diagram {
        border: 1px solid #E0E0E0;
        border-radius: 5px;
        padding: 10px;
        background-color: #FAFAFA;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">Regex to DFA, CFG & PDA Converter</h1>', unsafe_allow_html=True)

# Define the regular expressions that will be available in the select box
regex_options = {
    "(a+b)*": "(a+b)*",
    "a*b*": "a*b*",
    "(aba+bab) (a+b)*": "(aba+bab) (a+b)*",
    "(a+b+aa+bb)*": "(a+b+aa+bb)*",
    "(aba+bab)": "(aba+bab)",
    "(aba+bab) (a+b)*": "(aba+bab) (a+b)*",
    "(aba+bab) (a+b+ab+ba)*": "(aba+bab) (a+b+ab+ba)*",
    "(aba+bab)* (a+b)*": "(aba+bab)* (a+b)*",
    "(aba+bab) (a+b+aa+bb)*": "(aba+bab) (a+b+aa+bb)*",
    "(aba+bab) (a+b+aa)* (a+b)*": "(aba+bab) (a+b+aa)* (a+b)*",
    "(aba+bab) (a+b)* (baa)*": "(aba+bab) (a+b)* (baa)*",
}

# Add explanation tooltip
with st.expander("ℹ️ How to use this application"):
    st.markdown("""
    This application helps you understand how Regular Expressions are converted to different types of automata:
    
    1. **Select a Regular Expression** from the dropdown menu or use the predefined examples.
    2. **View the DFA** (Deterministic Finite Automaton) that represents the selected regex.
    3. **Enter a string** in the input field and click "Validate" to check if it's accepted by the DFA.
    4. **Watch the animation** showing how the DFA processes each character of your input string.
    5. **Explore the CFG and PDA** representations by expanding the sections below the diagram.
    
    **Key Components:**
    - **DFA** (Deterministic Finite Automaton): States with definite transitions for each input symbol.
    - **CFG** (Context-Free Grammar): A set of production rules that generate all valid strings.
    - **PDA** (Pushdown Automaton): An automaton with a stack to recognize context-free languages.
    """)


# Thompson's algorithm for NFA construction - optimized implementation
class NFAState:
    id_counter = 0
    
    def __init__(self, is_final=False):
        self.transitions = {}  # Dictionary to store transitions {symbol: [states]}
        self.epsilon_transitions = []  # List to store epsilon transitions
        self.is_final = is_final
        self.state_id = NFAState.id_counter
        NFAState.id_counter += 1

    def add_transition(self, symbol, state):
        if symbol in self.transitions:
            self.transitions[symbol].append(state)
        else:
            self.transitions[symbol] = [state]

    def add_epsilon_transition(self, state):
        self.epsilon_transitions.append(state)


class NFA:
    def __init__(self, start_state, end_state):
        self.start_state = start_state
        self.end_state = end_state
        end_state.is_final = True


@st.cache_data
def epsilon_closure(state, visited=None):
    """Calculate epsilon closure with caching for performance."""
    if visited is None:
        visited = set()

    if state in visited:
        return visited

    visited.add(state)
    for eps_state in state.epsilon_transitions:
        epsilon_closure(eps_state, visited)

    return visited


@st.cache_data
def regex_to_nfa(_regex):
    """Convert a regular expression to an NFA using Thompson's construction algorithm."""
    # Reset state counter for consistent state numbering
    NFAState.id_counter = 0
    
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
                    start = NFAState()
                    end = NFAState(is_final=True)
                    start.add_epsilon_transition(end)
                    return NFA(start, end), i

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
                else:
                    # Handle empty regex with Kleene star
                    start = NFAState()
                    end = NFAState(is_final=True)
                    start.add_epsilon_transition(end)
                    current_nfa = NFA(start, end)

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
    regex = _regex.replace(" ", "")

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


@st.cache_data
def nfa_to_dfa(nfa):
    """Convert an NFA to a DFA using the subset construction algorithm with performance optimizations."""
    alphabet = set()

    # Collect all symbols in the NFA using BFS for better performance
    queue = [nfa.start_state]
    visited = set()
    
    while queue:
        state = queue.pop(0)
        if state in visited:
            continue
            
        visited.add(state)
        
        for symbol in state.transitions:
            alphabet.add(symbol)
            for next_state in state.transitions[symbol]:
                if next_state not in visited:
                    queue.append(next_state)
        
        for eps_state in state.epsilon_transitions:
            if eps_state not in visited:
                queue.append(eps_state)

    # Start with epsilon closure of the start state
    start_closure = frozenset(epsilon_closure(nfa.start_state))

    # Map NFA state sets to DFA states
    dfa_states = {start_closure: 0}  # Start state is 0
    dfa_transitions = {}
    dfa_final_states = set()

    # Check if the start state is also a final state
    if any(state.is_final for state in start_closure):
        dfa_final_states.add(0)

    # Process queue more efficiently
    queue = [start_closure]
    while queue:
        current_states = queue.pop(0)
        current_dfa_state = dfa_states[current_states]

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
                
            next_states_frozen = frozenset(epsilon_states)

            # Add new DFA state if needed
            if next_states_frozen not in dfa_states:
                dfa_states[next_states_frozen] = len(dfa_states)
                queue.append(next_states_frozen)

                # Check if it's a final state
                if any(state.is_final for state in epsilon_states):
                    dfa_final_states.add(dfa_states[next_states_frozen])

            # Add transition to DFA
            dfa_transitions[(current_dfa_state, symbol)] = dfa_states[next_states_frozen]

    # Create the formal DFA structure
    dfa = {
        'states': set(range(len(dfa_states))),
        'alphabet': alphabet,
        'transitions': dfa_transitions,
        'start_state': 0,
        'final_states': dfa_final_states
    }

    return dfa


@st.cache_data
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
        for symbol in sorted(dfa['alphabet']):  # Sort for deterministic output
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


@st.cache_data
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
    for (state, symbol), next_state in sorted(dfa['transitions'].items()):  # Sort for deterministic output
        # Push the next state onto the stack
        pda['transitions'][(state, symbol, 'Z0')] = (next_state, [f"X{next_state}", 'Z0'])

        # For each stack symbol (except Z0)
        for stack_state in sorted(dfa['states']):  # Sort for deterministic output
            pda['transitions'][(state, symbol, f"X{stack_state}")] = (next_state, [f"X{next_state}", f"X{stack_state}"])

    # Add epsilon transitions for final states to pop the stack
    for final_state in sorted(dfa['final_states']):  # Sort for deterministic output
        pda['transitions'][(final_state, 'ε', 'Z0')] = (final_state, ['Z0'])
        for stack_state in sorted(dfa['states']):  # Sort for deterministic output
            pda['transitions'][(final_state, 'ε', f"X{stack_state}")] = (final_state, [])

    return pda


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


@st.cache_data
def visualize_dfa(dfa, highlight_states=None, highlight_transitions=None):
    """Visualize a DFA using networkx and matplotlib."""
    if highlight_states is None:
        highlight_states = []
    if highlight_transitions is None:
        highlight_transitions = []

    G = nx.DiGraph()

    # Add all states as nodes
    for state in dfa['states']:
        G.add_node(state, is_final=state in dfa['final_states'], is_start=state == dfa['start_state'])

    # Add transitions as edges
    for (state, symbol), next_state in sorted(dfa['transitions'].items()):  # Sort for consistent layouts
        # Check if an edge already exists
        if G.has_edge(state, next_state):
            # Update the label
            G[state][next_state]['label'] += f", {symbol}"
        else:
            G.add_edge(state, next_state, label=symbol)

    # Create a figure
    plt.figure(figsize=(12, 8))

    # Position the nodes
    pos = nx.spring_layout(G, seed=42)

    # Draw the graph
    node_colors = ['lightgreen' if node in highlight_states 
                   else ('lightblue' if G.nodes[node]['is_final'] else 'white')
                   for node in G.nodes]

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=700, node_color=node_colors, edgecolors='black')

    # Double circle for final states
    final_states = [state for state in dfa['states'] if state in dfa['final_states']]
    nx.draw_networkx_nodes(G, pos, nodelist=final_states, node_size=600, node_color='none', edgecolors='black')

    # Special marker for start state
    start_state = dfa['start_state']
    plt.annotate('', xy=pos[start_state], xytext=(pos[start_state][0] - 0.1, pos[start_state][1]),
                 arrowprops=dict(arrowstyle="->", color='black'))

    # Draw edges
    edges = G.edges()
    edge_colors = []

    for u, v in edges:
        if (u, v) in highlight_transitions:
            edge_colors.append('red')
        else:
            edge_colors.append('black')

    nx.draw_networkx_edges(G, pos, edgelist=edges, width=1.5, arrowsize=20, edge_color=edge_colors)

    # Add edge labels
    edge_labels = {(u, v): G[u][v]['label'] for u, v in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=12)

    # Add node labels (state numbers)
    node_labels = {node: f"q{node}" for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=12)

    plt.axis('off')
    plt.title("DFA Visualization", fontsize=16)

    return plt


def create_dfa_animation(dfa, input_string, states_visited):
    """Create an animation of the DFA processing the input string."""
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

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(12, 8))

    # Function to draw the current state of the DFA
    def draw_frame(i):
        ax.clear()

        # Determine highlight states and transitions
        highlight_states = [states_visited[min(i, len(states_visited) - 1)]]

        highlight_transitions = []
        if i > 0 and i < len(states_visited):
            highlight_transitions = [(states_visited[i - 1], states_visited[i])]

        # Draw nodes
        node_colors = [
            'lightgreen' if node in highlight_states else ('lightblue' if G.nodes[node]['is_final'] else 'white') for
            node in G.nodes]
        nx.draw_networkx_nodes(G, pos, node_size=700, node_color=node_colors, edgecolors='black', ax=ax)

        # Double circle for final states
        final_states = [state for state in dfa['states'] if state in dfa['final_states']]
        nx.draw_networkx_nodes(G, pos, nodelist=final_states, node_size=600, node_color='none', edgecolors='black',
                               ax=ax)

        # Special marker for start state
        start_state = dfa['start_state']
        ax.annotate('', xy=pos[start_state], xytext=(pos[start_state][0] - 0.1, pos[start_state][1]),
                    arrowprops=dict(arrowstyle="->", color='black'))

        # Draw edges
        edges = G.edges()
        edge_colors = []

        for u, v in edges:
            if (u, v) in highlight_transitions:
                edge_colors.append('red')
            else:
                edge_colors.append('black')

        nx.draw_networkx_edges(G, pos, edgelist=edges, width=1.5, arrowsize=20, edge_color=edge_colors, ax=ax)

        # Add edge labels
        edge_labels = {(u, v): G[u][v]['label'] for u, v in G.edges()}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=12, ax=ax)

        # Add node labels (state numbers)
        node_labels = {node: f"q{node}" for node in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=12, ax=ax)

        # Display the current character being processed
        if i > 0 and i < len(input_string) + 1:
            char = input_string[i - 1]
            current_state = states_visited[min(i - 1, len(states_visited) - 1)]
            next_state = states_visited[min(i, len(states_visited) - 1)]

            # If there's a transition, show the symbol on the edge
            if current_state != next_state:
                edge_label = G[current_state][next_state]['label']
                ax.set_title(f"Processing: '{char}' | Transition: q{current_state} --{char}--> q{next_state}",
                             fontsize=16)
            else:
                ax.set_title(f"Processing: '{char}' | No valid transition from q{current_state}", fontsize=16)
        elif i == 0:
            ax.set_title(f"Start: at state q{states_visited[0]}", fontsize=16)
        else:
            result = "Accepted" if states_visited[-1] in dfa['final_states'] else "Rejected"
            ax.set_title(f"Finished: String {result}", fontsize=16)

        ax.axis('off')

    # Create the animation
    anim = animation.FuncAnimation(fig, draw_frame, frames=len(states_visited) + 1, interval=1000, repeat=True)

    plt.close(fig)  # Don't display the figure, just return the animation

    return anim


# Function to display the regular expression to automata conversion
def display_regex_conversion(regex, input_string=None):
    with st.spinner(f"Generating automata for {regex}..."):
        st.markdown(f'<h2 class="sub-header">Deterministic Finite Automaton for: {regex}</h2>', unsafe_allow_html=True)

        try:
            # Convert regex to NFA
            nfa = regex_to_nfa(regex)

            # Convert NFA to DFA
            dfa = nfa_to_dfa(nfa)

            # Store the dfa in session state for validation
            st.session_state.current_dfa = dfa

            # Generate CFG and PDA
            cfg = dfa_to_cfg(dfa)
            pda = dfa_to_pda(dfa)

            # Visualize DFA
            if input_string:
                states_visited, is_valid = simulate_dfa(dfa, input_string)
                st.write(f"Entered String: `{input_string}`")

                # Create animation
                anim = create_dfa_animation(dfa, input_string, states_visited)

                # Display animation
                html_anim = HTML(anim.to_jshtml()).data

                # Show the animation inside a styled container
                st.markdown('<div class="state-diagram">', unsafe_allow_html=True)
                st.components.v1.html(html_anim, height=600)
                st.markdown('</div>', unsafe_allow_html=True)

                # Show validation result
                if is_valid:
                    st.markdown(f'<div class="success-message">✅ The string \'{input_string}\' is valid for the DFA.</div>',
                                unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="error-message">❌ The string \'{input_string}\' is NOT valid for the DFA.</div>',
                                unsafe_allow_html=True)
            else:
                # Just show the static DFA visualization
                st.markdown('<div class="state-diagram">', unsafe_allow_html=True)
                fig = visualize_dfa(dfa)
                st.pyplot(fig)
                st.markdown('</div>', unsafe_allow_html=True)

            # Information about the DFA
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"Number of states: {len(dfa['states'])}")
            with col2:
                st.info(f"Alphabet: {', '.join(sorted(dfa['alphabet']))}")

            # Show CFG and PDA in expanders
            with st.expander("Context-Free Grammar (CFG) Representation"):
                st.markdown("### Productions")
                st.markdown(f"Start Symbol: `{cfg['start_symbol']}`")
                
                # Format productions nicely
                for nt in sorted(cfg['productions'].keys()):
                    productions = cfg['productions'][nt]
                    production_str = " | ".join(productions) if productions else "ε"
                    st.markdown(f"`{nt} → {production_str}`")

            with st.expander("Pushdown Automaton (PDA) Representation"):
                st.markdown("### States")
                st.markdown(f"States: {', '.join([f'q{s}' for s in sorted(pda['states'])])}")
                st.markdown(f"Start State: q{pda['start_state']}")
                st.markdown(f"Final States: {', '.join([f'q{s}' for s in sorted(pda['final_states'])])}")

                st.markdown("### Alphabets")
                st.markdown(f"Input Alphabet: {', '.join(sorted(pda['input_alphabet']))}")
                st.markdown(f"Stack Alphabet: {', '.join(sorted(pda['stack_alphabet']))}")

                st.markdown("### Transitions")
                # Format transitions in a table for better readability
                transitions_data = []
                for (state, symbol, stack_top), (next_state, new_stack) in sorted(pda['transitions'].items()):
                    new_stack_str = ''.join(new_stack) if new_stack else 'ε'
                    transition = {
                        "From State": f"q{state}",
                        "Input Symbol": symbol if symbol != 'ε' else 'ε',
                        "Stack Top": stack_top,
                        "To State": f"q{next_state}",
                        "New Stack": new_stack_str
                    }
                    transitions_data.append(transition)
                
                if transitions_data:
                    st.dataframe(pd.DataFrame(transitions_data))
                else:
                    st.write("No transitions defined.")
                    
        except Exception as e:
            st.error(f"Error processing the regular expression: {str(e)}")
            st.warning("Please try a different regular expression or check the syntax.")


# Initialize session state for DFA
if 'current_dfa' not in st.session_state:
    st.session_state.current_dfa = None

# Main app layout
col1, col2 = st.columns([3, 1])

with col1:
    st.markdown('<h2 class="sub-header">Select a Regular Expression</h2>', unsafe_allow_html=True)
    selected_regex = st.selectbox("", list(regex_options.keys()), format_func=lambda x: regex_options[x], label_visibility="collapsed")

with col2:
    custom_regex = st.checkbox("Use custom regex", value=False)

if custom_regex:
    user_regex = st.text_input("Enter your regular expression:", placeholder="e.g., (a+b)*aba")
    if user_regex:
        selected_regex = user_regex

# Display the conversion for the selected regex
display_regex_conversion(selected_regex)

# Section for string validation
st.markdown('<h2 class="sub-header">Enter a string to check its validity for displayed DFA</h2>', unsafe_allow_html=True)

col1, col2 = st.columns([3, 1])
with col1:
    input_string = st.text_input("", placeholder="Enter a string (e.g., abababba)", label_visibility="collapsed")
with col2:
    validate_button = st.button("Validate", use_container_width=True)

if validate_button:
    if input_string:
        # Display the conversion with animation for the input string
        display_regex_conversion(selected_regex, input_string)
    else:
        st.error("Please enter a string to validate.")

# Add examples and usage guide at the bottom
with st.expander("String Examples"):
    st.markdown("""
    ### Sample strings to try:
    - For `(a+b)*`: Try `ababba`, `aaa`, `bbb`, or empty string
    - For `a*b*`: Try `aaabbb`, `aaa`, `bbb`, or empty string
    - For `(aba+bab)`: Try `aba` or `bab` (only these two strings are valid)
    - For complex patterns, experiment with different combinations!
    """)

# Footer with additional resources
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>Created with Streamlit • <a href="https://github.com/yourusername/regex-automata" target="_blank">View source code</a></p>
</div>
""", unsafe_allow_html=True)