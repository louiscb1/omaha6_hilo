import streamlit as st
import random
from card_utils import create_deck, shuffle_deck, deal_cards
from hand_evaluator import evaluate_high_hand, evaluate_low_hand
from itertools import combinations
from tqdm import tqdm

# Helper functions
def parse_hand_dropdowns(selected_cards):
    return [card for card in selected_cards if card != ""]

def parse_board_string(board_str):
    valid_ranks = "23456789TJQKA"
    valid_suits = "cdhs"
    board_str = board_str.strip()
    if board_str.lower() == "none" or board_str == "":
        return []
    cards = [board_str[i:i+2] for i in range(0, len(board_str), 2)]
    if len(cards) > 5:
        raise ValueError("Board can have at most 5 cards.")
    for card in cards:
        if len(card) != 2 or card[0] not in valid_ranks or card[1] not in valid_suits:
            raise ValueError(f"Invalid board card format: '{card}'")
    return cards

def check_for_duplicate_cards(all_player_hands, board_cards):
    seen = set()
    for hand in all_player_hands:
        if hand is None:
            continue
        for card in hand:
            if card in seen:
                raise ValueError(f"Duplicate card detected: {card}")
            seen.add(card)
    for card in board_cards:
        if card in seen:
            raise ValueError(f"Duplicate card detected between player hands and board: {card}")
        seen.add(card)

def run_simulation(player_hands_input, board_input, num_sims):
    NUM_PLAYERS = 2
    high_wins = [0 for _ in range(NUM_PLAYERS)]
    low_wins = [0 for _ in range(NUM_PLAYERS)]
    scoops = [0 for _ in range(NUM_PLAYERS)]

    for sim in tqdm(range(num_sims)):
        deck = create_deck()
        shuffle_deck(deck)

        # Deal player hands
        player_hands = []
        for i in range(NUM_PLAYERS):
            if player_hands_input[i]:
                hand = player_hands_input[i]
                for card in hand:
                    deck.remove(card)
                player_hands.append(hand)
            else:
                player_hand = deal_cards(deck, 6)
                player_hands.append(player_hand)

        # Prepare board
        board = board_input.copy()
        for card in board:
            deck.remove(card)
        num_to_deal = 5 - len(board)
        if num_to_deal > 0:
            board += deal_cards(deck, num_to_deal)

        if sim == 0:
            check_for_duplicate_cards(player_hands, board)

        # Evaluate hands
        high_scores = []
        low_scores = []

        for player_hand in player_hands:
            hole_combos = combinations(player_hand, 2)
            board_combos = combinations(board, 3)

            best_high = None
            best_low = None

            for hole in hole_combos:
                for board3 in board_combos:
                    full_hand = list(hole) + list(board3)
                    high_score = evaluate_high_hand(full_hand)
                    low_score = evaluate_low_hand(full_hand)

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

        # Determine Low winner
        valid_lows = [score for score in low_scores if score is not None]
        min_low = None
        if valid_lows:
            min_low = min(valid_lows)
            low_winners = [i for i, score in enumerate(low_scores) if score == min_low]
            for w in low_winners:
                low_wins[w] += 1 / len(low_winners)

        # Determine scoops
        for i in range(NUM_PLAYERS):
            if min_low is not None and \
               high_scores[i] == max_high and low_scores[i] == min_low and \
               high_scores.count(max_high) == 1 and low_scores.count(min_low) == 1:
                scoops[i] += 1

    return high_wins, low_wins, scoops

# Streamlit UI
st.set_page_config(page_title="Omaha 6 Hi/Lo Equity Calculator", page_icon="🃏")

st.title("🃏 Omaha 6 Hi/Lo Equity Calculator")
st.markdown(
    """
    Select two 6-card Omaha hands and an optional partial board.  
    Choose number of simulations and run the calculator.  
    Results will show High %, Low %, and Scoops % for each player.
    """
)

# Prepare card list
ranks = "23456789TJQKA"
suits = "cdhs"
deck = [r + s for r in ranks for s in suits]
deck.insert(0, "")  # Empty option to allow unselected slots

# Session state for Player hands
if "p1_cards" not in st.session_state:
    st.session_state.p1_cards = [""] * 6
if "p2_cards" not in st.session_state:
    st.session_state.p2_cards = [""] * 6

# Smart dropdown filtering
def get_available_cards(exclude_cards):
    return [card for card in deck if card not in exclude_cards]

# Player 1
st.header("Player 1 Hand")
cols1 = st.columns(6)
p1_selected = []
for i in range(6):
    available_cards = get_available_cards(p1_selected + st.session_state.p2_cards)
    st.session_state.p1_cards[i] = cols1[i].selectbox(
        f"P1 Card {i+1}",
        available_cards,
        index=available_cards.index(st.session_state.p1_cards[i]) if st.session_state.p1_cards[i] in available_cards else 0
    )
    if st.session_state.p1_cards[i] != "":
        p1_selected.append(st.session_state.p1_cards[i])

if st.button("🎲 Randomize Player 1 Hand"):
    st.session_state.p1_cards = random.sample(deck[1:], 6)
    st.experimental_rerun()

# Player 2
st.header("Player 2 Hand")
cols2 = st.columns(6)
p2_selected = []
for i in range(6):
    available_cards = get_available_cards(p2_selected + st.session_state.p1_cards)
    st.session_state.p2_cards[i] = cols2[i].selectbox(
        f"P2 Card {i+1}",
        available_cards,
        index=available_cards.index(st.session_state.p2_cards[i]) if st.session_state.p2_cards[i] in available_cards else 0
    )
    if st.session_state.p2_cards[i] != "":
        p2_selected.append(st.session_state.p2_cards[i])

if st.button("🎲 Randomize Player 2 Hand"):
    st.session_state.p2_cards = random.sample(deck[1:], 6)
    st.experimental_rerun()

# Board + Run
st.header("Board")
board_str = st.text_input("Partial Board (optional, e.g. Ah2s3c)", value="")

num_sims = st.slider("Number of Simulations", min_value=1000, max_value=500_000, value=100_000, step=1000)

run_clicked = st.button("🚀 Run Simulation")
reset_clicked = st.button("🔄 Reset Inputs")

if reset_clicked:
    st.session_state.p1_cards = [""] * 6
    st.session_state.p2_cards = [""] * 6
    st.experimental_rerun()

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

        board_input = parse_board_string(board_str)

        # Run simulation
        with st.spinner("Running simulations..."):
            high_wins, low_wins, scoops = run_simulation(player_hands_input, board_input, num_sims)

        # Display results
        st.subheader("Results")
        for i in range(2):
            st.write(f"**Player {i+1}:**")
            st.write(f"High wins: {100 * high_wins[i] / num_sims:.2f}%")
            st.write(f"Low wins: {100 * low_wins[i] / num_sims:.2f}%")
            st.write(f"Scoops:    {100 * scoops[i] / num_sims:.2f}%")
    except Exception as e:
        st.error(f"Error: {e}")
