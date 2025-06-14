import streamlit as st
from card_utils import create_deck, shuffle_deck, deal_cards
from hand_evaluator import evaluate_high_hand, evaluate_low_hand
from itertools import combinations
import pandas as pd
import io
from datetime import datetime
import matplotlib.pyplot as plt
st.set_page_config(layout="wide")

ranks = "AKQJT98765432"
suits = "shdc"
full_deck = [r + s for s in suits for r in ranks]

# Session state initialization
if "p1_hand" not in st.session_state:
    st.session_state.p1_hand = []
if "p2_hand" not in st.session_state:
    st.session_state.p2_hand = []
if "board_cards" not in st.session_state:
    st.session_state.board_cards = []
if "selected_cards" not in st.session_state:
    st.session_state.selected_cards = set()
if "selection_target" not in st.session_state:
    st.session_state.selection_target = "P1"

def suit_to_emoji(card):
    suit_map = {'s': ('♠', 'black'), 'h': ('♥', 'red'), 'd': ('♦', 'red'), 'c': ('♣', 'black')}
    symbol, color = suit_map[card[1]]
    return f"<span style='color:{color}; font-weight:bold'>{card[0]}{symbol}</span>"

def render_card_selector():
    st.subheader("Card Selector")
    st.radio("Assign cards to:", ["P1", "P2", "Board"], key="selection_target")
    for suit in suits:
     cols = st.columns(13)
     for idx, rank in enumerate(ranks):
        card = rank + suit
        label = f"{rank}♠" if suit == 's' else f"{rank}♥" if suit == 'h' else f"{rank}♦" if suit == 'd' else f"{rank}♣"
        style = f"color: {'red' if suit in 'hd' else 'black'}; font-weight:bold"

        with cols[idx]:
            if card in st.session_state.selected_cards:
                st.markdown(f"<div style='{style}; opacity: 0.4'>{label}</div>", unsafe_allow_html=True)
            else:
                if st.button(label, key=card):
                    assign_card(card)


def assign_card(card):
    target = st.session_state.selection_target
    if target == "P1" and len(st.session_state.p1_hand) < 6:
        st.session_state.p1_hand.append(card)
    elif target == "P2" and len(st.session_state.p2_hand) < 6:
        st.session_state.p2_hand.append(card)
    elif target == "Board" and len(st.session_state.board_cards) < 5:
        st.session_state.board_cards.append(card)
    else:
        return
    st.session_state.selected_cards.add(card)

def reset_inputs():
    st.session_state.p1_hand = []
    st.session_state.p2_hand = []
    st.session_state.board_cards = []
    st.session_state.selected_cards = set()

# Main interface
st.title("Omaha 6 HiLo Equity Calculator")
render_card_selector()

st.markdown("**P1 Hand:** " + " ".join([suit_to_emoji(c) for c in st.session_state.p1_hand]), unsafe_allow_html=True)
st.markdown("**P2 Hand:** " + " ".join([suit_to_emoji(c) for c in st.session_state.p2_hand]), unsafe_allow_html=True)
st.markdown("**Board:** " + " ".join([suit_to_emoji(c) for c in st.session_state.board_cards]), unsafe_allow_html=True)

st.button("Reset Inputs", on_click=reset_inputs)

st.header("Simulation Settings")
num_sims = st.number_input("Number of Simulations", min_value=100, max_value=1000000, value=10000, step=1000)
run_clicked = st.button("Run Simulation")


def run_simulation(player_hands_input, board, num_simulations):
    win_hi = [0, 0]
    tie_hi = [0, 0]
    win_lo = [0, 0]
    tie_lo = [0, 0]
    example_board = []

    for _ in range(num_simulations):
        deck = create_deck()
        for card in player_hands_input[0] + player_hands_input[1] + board:
            if card in deck:
                deck.remove(card)
        shuffle_deck(deck)
        sim_board = board.copy()
        if (5 - len(board)) > 0:
            sim_board += deal_cards(deck, 5 - len(board))
        if not example_board:
            example_board = sim_board.copy()

        high_scores = []
        low_scores = []

        for player_hand in player_hands_input:
            hole_combos = combinations(player_hand, 2)
            board_combos = combinations(sim_board, 3)
            best_high, best_low = None, None
            for hole in hole_combos:
                for board3 in board_combos:
                    full_hand = list(hole) + list(board3)
                    high_score = evaluate_high_hand(full_hand)
                    low_score = evaluate_low_hand(full_hand, sim_board)
                    if best_high is None or high_score > best_high:
                        best_high = high_score
                    if low_score is not None:
                        if best_low is None or low_score < best_low:
                            best_low = low_score
            high_scores.append(best_high)
            low_scores.append(best_low)

        max_high = max(high_scores)
        high_winners = [i for i, score in enumerate(high_scores) if score == max_high]
        for i in high_winners:
            if len(high_winners) == 1:
                win_hi[i] += 1
            else:
                tie_hi[i] += 1

        valid_lows = [s for s in low_scores if s is not None]
        if valid_lows:
            min_low = min(valid_lows)
            low_winners = [i for i, score in enumerate(low_scores) if score == min_low]
            for i in low_winners:
                if len(low_winners) == 1:
                    win_lo[i] += 1
                else:
                    tie_lo[i] += 1

    return win_hi, tie_hi, win_lo, tie_lo, example_board

if run_clicked:
    try:
        player_hands_input = [st.session_state.p1_hand, st.session_state.p2_hand]
        for hand in player_hands_input:
            if len(hand) != 6:
                raise ValueError("Each player must have exactly 6 cards.")
        board_input = st.session_state.board_cards
        if len(board_input) not in [0, 3, 5]:
            raise ValueError("Board must be 0, 3, or 5 cards.")

        with st.spinner("Running simulations..."):
            win_hi, tie_hi, win_lo, tie_lo, example_board = run_simulation(player_hands_input, board_input, num_sims)

        st.subheader("Results")
        for i in range(2):
            st.write(f"**Player {i+1}:**")
            st.write(f"Win High: {100 * win_hi[i] / num_sims:.2f}%")
            st.write(f"Tie High: {100 * tie_hi[i] / num_sims:.2f}%")
            st.write(f"Win Low:  {100 * win_lo[i] / num_sims:.2f}%")
            st.write(f"Tie Low:  {100 * tie_lo[i] / num_sims:.2f}%")

        st.subheader("Results Bar Chart")
        labels = ["Win High", "Tie High", "Win Low", "Tie Low"]
        p1_values = [100 * win_hi[0] / num_sims, 100 * tie_hi[0] / num_sims, 100 * win_lo[0] / num_sims, 100 * tie_lo[0] / num_sims]
        p2_values = [100 * win_hi[1] / num_sims, 100 * tie_hi[1] / num_sims, 100 * win_lo[1] / num_sims, 100 * tie_lo[1] / num_sims]

        x = range(len(labels))
        width = 0.35

        fig, ax = plt.subplots()
        ax.bar([i - width/2 for i in x], p1_values, width, label='Player 1')
        ax.bar([i + width/2 for i in x], p2_values, width, label='Player 2')

        ax.set_ylabel('Percentage (%)')
        ax.set_title('Simulation Results')
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels)
        ax.legend()

        st.pyplot(fig)

        st.subheader("Example Board")
        st.markdown(" ".join([suit_to_emoji(card) for card in example_board]), unsafe_allow_html=True)

        player1_hand_str = " ".join(st.session_state.p1_hand)
        player2_hand_str = " ".join(st.session_state.p2_hand)
        board_str = " ".join(board_input)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        data = {
            "Timestamp": [timestamp],
            "Player 1 Hand": [player1_hand_str],
            "Player 2 Hand": [player2_hand_str],
            "Board": [board_str],
            "Number of Simulations": [num_sims],
            "P1 Win High %": [100 * win_hi[0] / num_sims],
            "P1 Tie High %": [100 * tie_hi[0] / num_sims],
            "P1 Win Low %": [100 * win_lo[0] / num_sims],
            "P1 Tie Low %": [100 * tie_lo[0] / num_sims],
            "P2 Win High %": [100 * win_hi[1] / num_sims],
            "P2 Tie High %": [100 * tie_hi[1] / num_sims],
            "P2 Win Low %": [100 * win_lo[1] / num_sims],
            "P2 Tie Low %": [100 * tie_lo[1] / num_sims]
        }

        df = pd.DataFrame(data)
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue()

        st.download_button(
            label="📥 Save Results to CSV",
            data=csv_data,
            file_name=f"omaha6_hilo_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"Error: {e}")