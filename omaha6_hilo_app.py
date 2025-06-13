import streamlit as st
from card_utils import create_deck, shuffle_deck, deal_cards
from hand_evaluator import evaluate_high_hand, evaluate_low_hand
from itertools import combinations
import pandas as pd
import io
from datetime import datetime

# Helper functions
def parse_hand_dropdowns(selected_cards):
    return [card for card in selected_cards if card != ""]

def parse_board_dropdowns(selected_cards):
    return [card for card in selected_cards if card != ""]

def suit_to_emoji(card):
    suit = card[1]
    suit_map = {
        's': '♠',
        'h': '♥',
        'd': '♦',
        'c': '♣'
    }
    return card[0] + suit_map.get(suit, '')

def get_available_cards(selected_cards):
    all_ranks = "23456789TJQKA"
    all_suits = "shdc"
    full_deck = [r + s for r in all_ranks for s in all_suits]
    return [""] + [card for card in full_deck if card not in selected_cards]

# Run Simulation function
def run_simulation(player_hands_input, board, num_simulations):
    high_wins = [0, 0]
    low_wins = [0, 0]
    scoops = [0, 0]

    example_board = []

    for _ in range(num_simulations):
        deck = create_deck()

        # Remove known player cards and board cards from deck
        for card in player_hands_input[0] + player_hands_input[1] + board:
            if card in deck:
                deck.remove(card)

        shuffle_deck(deck)

        # Complete the board
        num_to_deal = 5 - len(board)
        sim_board = board.copy()
        if num_to_deal > 0:
            sim_board += deal_cards(deck, num_to_deal)

        if not example_board:
            example_board = sim_board.copy()

        high_scores = []
        low_scores = []

        for player_hand in player_hands_input:
            # All combos of 2 hole cards
            hole_combos = combinations(player_hand, 2)
            # All combos of 3 board cards
            board_combos = combinations(sim_board, 3)

            best_high = None
            best_low = None

            for hole in hole_combos:
                for board3 in board_combos:
                    full_hand = list(hole) + list(board3)
                    high_score = evaluate_high_hand(full_hand)
                    low_score = evaluate_low_hand(full_hand, sim_board)

                    if (best_high is None) or (high_score > best_high):
                        best_high = high_score
                    if low_score is not None:
                        if (best_low is None) or (low_score < best_low):
                            best_low = low_score

            high_scores.append(best_high)
            low_scores.append(best_low)

        # Determine High winner
        max_high = max(high_scores)
        high_winners = [i for i, score in enumerate(high_scores) if score == max_high]
        for w in high_winners:
            high_wins[w] += 1 / len(high_winners)

        # Determine Low winner (if any Low)
        valid_lows = [score for score in low_scores if score is not None]
        if valid_lows:
            min_low = min(valid_lows)
            low_winners = [i for i, score in enumerate(low_scores) if score == min_low]
            for w in low_winners:
                low_wins[w] += 1 / len(low_winners)

        # Determine scoops (player who wins both High and Low alone)
        for i in range(2):
            if high_scores[i] == max_high and (low_scores[i] == min_low if valid_lows else True) and \
               high_scores.count(max_high) == 1 and (low_scores.count(min_low) == 1 if valid_lows else True):
                scoops[i] += 1

    return high_wins, low_wins, scoops, example_board

# Streamlit UI
st.title("Omaha 6 HiLo Equity Calculator (PRO)")

# Player 1 Hand
st.header("Player 1 Hand")
if "p1_cards" not in st.session_state:
    st.session_state.p1_cards = [""] * 6

cols_p1 = st.columns(6)
p1_selected = []
for i in range(6):
    available_cards = get_available_cards(p1_selected + st.session_state.p2_cards + parse_board_dropdowns(st.session_state.board_cards) if "board_cards" in st.session_state else [])
    st.session_state.p1_cards[i] = cols_p1[i].selectbox(
        f"P1 Card {i+1}",
        available_cards,
        index=available_cards.index(st.session_state.p1_cards[i]) if st.session_state.p1_cards[i] in available_cards else 0
    )
    if st.session_state.p1_cards[i] != "":
        p1_selected.append(st.session_state.p1_cards[i])

# Player 2 Hand
st.header("Player 2 Hand")
if "p2_cards" not in st.session_state:
    st.session_state.p2_cards = [""] * 6

cols_p2 = st.columns(6)
p2_selected = []
for i in range(6):
    available_cards = get_available_cards(p2_selected + st.session_state.p1_cards + parse_board_dropdowns(st.session_state.board_cards) if "board_cards" in st.session_state else [])
    st.session_state.p2_cards[i] = cols_p2[i].selectbox(
        f"P2 Card {i+1}",
        available_cards,
        index=available_cards.index(st.session_state.p2_cards[i]) if st.session_state.p2_cards[i] in available_cards else 0
    )
    if st.session_state.p2_cards[i] != "":
        p2_selected.append(st.session_state.p2_cards[i])

# Board input — Safe smart dropdowns
st.header("Board")
if "board_cards" not in st.session_state:
    st.session_state.board_cards = [""] * 5

cols_board = st.columns(5)
board_selected = []
for i in range(5):
    available_cards = get_available_cards(board_selected + st.session_state.p1_cards + st.session_state.p2_cards)
    st.session_state.board_cards[i] = cols_board[i].selectbox(
        f"Board Card {i+1}",
        available_cards,
        index=available_cards.index(st.session_state.board_cards[i]) if st.session_state.board_cards[i] in available_cards else 0
    )
    if st.session_state.board_cards[i] != "":
        board_selected.append(st.session_state.board_cards[i])

# Check valid board input
valid_board_input = len(board_selected) in [0, 3]

# Show warning if invalid
if not valid_board_input:
    st.warning("Please select either 0 cards or exactly 3 board cards for runout simulation.")

# Simulation Settings
st.header("Simulation Settings")
num_sims = st.number_input("Number of Simulations", min_value=100, max_value=1000000, value=10000, step=1000)

# Run button — only enabled if valid input
run_clicked = st.button("Run Simulation", disabled=not valid_board_input)

# Reset button
reset_clicked = st.button("Reset Inputs")

if reset_clicked:
    st.session_state.p1_cards = [""] * 6
    st.session_state.p2_cards = [""] * 6
    st.session_state.board_cards = [""] * 5

# Run Simulation
if run_clicked:
    try:
        # Parse hands
        player_hands_input = [
            parse_hand_dropdowns(st.session_state.p1_cards),
            parse_hand_dropdowns(st.session_state.p2_cards)
        ]

        # Validate hands
        for hand in player_hands_input:
            if len(hand) not in [0, 6]:
                raise ValueError("Each player hand must be either empty or exactly 6 unique cards.")

        # Parse Board
        board_input = parse_board_dropdowns(st.session_state.board_cards)
        board_str = " ".join(board_input)

        # Run simulation
        with st.spinner("Running simulations..."):
            high_wins, low_wins, scoops, example_board = run_simulation(player_hands_input, board_input, num_sims)

        # Display results
        st.subheader("Results")
        for i in range(2):
            st.write(f"**Player {i+1}:**")
            st.write(f"High wins: {100 * high_wins[i] / num_sims:.2f}%")
            st.write(f"Low wins: {100 * low_wins[i] / num_sims:.2f}%")
            st.write(f"Scoops:    {100 * scoops[i] / num_sims:.2f}%")

        # Show example board
        st.subheader("Example Board from Simulation")
        st.write(" ".join([suit_to_emoji(card) for card in example_board]))

        # Prepare CSV data
        player1_hand_str = " ".join(parse_hand_dropdowns(st.session_state.p1_cards))
        player2_hand_str = " ".join(parse_hand_dropdowns(st.session_state.p2_cards))

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        data = {
            "Timestamp": [timestamp],
            "Player 1 Hand": [player1_hand_str],
            "Player 2 Hand": [player2_hand_str],
            "Board": [board_str],
            "Number of Simulations": [num_sims],
            "P1 High Wins %": [100 * high_wins[0] / num_sims],
            "P1 Low Wins %": [100 * low_wins[0] / num_sims],
            "P1 Scoops %": [100 * scoops[0] / num_sims],
            "P2 High Wins %": [100 * high_wins[1] / num_sims],
            "P2 Low Wins %": [100 * low_wins[1] / num_sims],
            "P2 Scoops %": [100 * scoops[1] / num_sims],
        }

        df = pd.DataFrame(data)

        # Convert to CSV
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue()

        # Download button
        st.download_button(
            label="📥 Save Results to CSV",
            data=csv_data,
            file_name=f"omaha6_hilo_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"Error: {e}")

