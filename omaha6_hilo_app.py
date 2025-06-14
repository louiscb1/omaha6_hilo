# omaha6_hilo_app.py (Updated)
import streamlit as st
from card_utils import create_deck, shuffle_deck, deal_cards
from hand_evaluator import evaluate_high_hand, evaluate_low_hand
from itertools import combinations
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt

# -- UI Helpers --
def parse_hand(cards):
    return [card for card in cards if card]

def suit_to_emoji(card):
    suit_map = {'s': '♠', 'h': '♥', 'd': '♦', 'c': '♣'}
    color_map = {'s': 'black', 'c': 'black', 'h': 'red', 'd': 'red'}
    return f"<span style='color:{color_map.get(card[1], 'black')}'>{card[0]}{suit_map.get(card[1], '')}</span>"

# -- Game Rules --
def qualifies_low(hand):
    if hand is None:
        return False
    return all(card[0] in "2345678A" for card in hand) and len(set(c[0] for c in hand)) == 5

def is_nut_low(score):
    return score == (5, 4, 3, 2, 1)

# -- Simulation Engine --
def run_simulation(p1, p2, board, num_sims):
    win_hi = [0, 0]
    tie_hi = [0, 0]
    win_lo = [0, 0]
    tie_lo = [0, 0]
    scoops = [0, 0]
    equity = [0.0, 0.0]
    nut_hi = [0, 0]
    nut_lo = [0, 0]
    example_board = []

    for _ in range(num_sims):
        deck = create_deck()
        known = p1 + p2 + board
        for card in known:
            if card in deck:
                deck.remove(card)
        shuffle_deck(deck)

        sim_board = board + deal_cards(deck, 5 - len(board))
        if not example_board:
            example_board = sim_board.copy()

        hi_scores, lo_scores, lo_hands = [], [], []
        for hand in [p1, p2]:
            best_hi = None
            best_lo = None
            best_lo_hand = None
            for hole in combinations(hand, 2):
                for board3 in combinations(sim_board, 3):
                    full = list(hole) + list(board3)
                    hi = evaluate_high_hand(full)
                    lo = evaluate_low_hand(full, sim_board)
                    if best_hi is None or hi > best_hi:
                        best_hi = hi
                    if lo is not None:
                        if best_lo is None or lo < best_lo:
                            best_lo = lo
                            best_lo_hand = full
            hi_scores.append(best_hi)
            lo_scores.append(best_lo)
            lo_hands.append(best_lo_hand)

        max_hi = max(hi_scores)
        hi_winners = [i for i, s in enumerate(hi_scores) if s == max_hi]
        for i in hi_winners:
            if len(hi_winners) == 1:
                win_hi[i] += 1
            else:
                tie_hi[i] += 1

        lo_winners = []
        valid_lows = [s for s in lo_scores if s is not None and qualifies_low(lo_hands[i])]
        if valid_lows:
            min_lo = min(valid_lows)
            lo_winners = [i for i, s in enumerate(lo_scores) if s == min_lo and qualifies_low(lo_hands[i])]
            for i in lo_winners:
                if len(lo_winners) == 1:
                    win_lo[i] += 1
                else:
                    tie_lo[i] += 1

        # Equity split
        pot_share = [0.0, 0.0]
        for i in hi_winners:
            pot_share[i] += 0.5 / len(hi_winners)
        if valid_lows:
            for i in lo_winners:
                pot_share[i] += 0.5 / len(lo_winners)
        for i in [0, 1]:
            equity[i] += pot_share[i]

        # Scoops
        for i in [0, 1]:
            if len(hi_winners) == 1 and len(lo_winners) == 1:
                if hi_winners[0] == i and lo_winners[0] == i:
                    scoops[i] += 1

        # Nut tracking
        if hi_scores[0] > hi_scores[1]:
            nut_hi[0] += 1
        elif hi_scores[1] > hi_scores[0]:
            nut_hi[1] += 1
        if lo_scores[0] and is_nut_low(lo_scores[0]):
            nut_lo[0] += 1
        if lo_scores[1] and is_nut_low(lo_scores[1]):
            nut_lo[1] += 1

    return {
        "win_hi": win_hi, "tie_hi": tie_hi,
        "win_lo": win_lo, "tie_lo": tie_lo,
        "scoops": scoops, "equity": equity,
        "nut_hi": nut_hi, "nut_lo": nut_lo,
        "example_board": example_board
    }

# -- Streamlit UI --
st.title("Omaha 6 HiLo Equity Calculator")

# Store player hands and board in session state
for key in ["p1_cards", "p2_cards", "board_cards"]:
    if key not in st.session_state:
        st.session_state[key] = [""] * (6 if "p" in key else 5)

def get_card_pool():
    ranks = "23456789TJQKA"
    suits = "shdc"
    return [r + s for r in ranks for s in suits]

def render_card_picker(label, card_state, opponent_cards, board_cards):
    all_cards = get_card_pool()
    selected_cards = opponent_cards + board_cards
    st.markdown(f"**{label}**")
    cols = st.columns(len(card_state))
    for i in range(len(card_state)):
        available = [""] + [c for c in all_cards if c not in selected_cards and c not in card_state if c != card_state[i]]
        card_state[i] = cols[i].selectbox(f"{label} {i+1}", options=available, index=available.index(card_state[i]) if card_state[i] in available else 0)

render_card_picker("Player 1", st.session_state.p1_cards, st.session_state.p2_cards, st.session_state.board_cards)
render_card_picker("Player 2", st.session_state.p2_cards, st.session_state.p1_cards, st.session_state.board_cards)
render_card_picker("Board", st.session_state.board_cards, st.session_state.p1_cards + st.session_state.p2_cards, [])

num_sims = st.number_input("# Simulations", 1000, 100000, 10000, step=1000)

if st.button("Run Simulation"):
    p1_hand = parse_hand(st.session_state.p1_cards)
    p2_hand = parse_hand(st.session_state.p2_cards)
    board_cards = parse_hand(st.session_state.board_cards)

    results = run_simulation(p1_hand, p2_hand, board_cards, num_sims)

    st.subheader("Results")
    for i in [0, 1]:
        st.write(f"**Player {i+1}**")
        st.write(f"Win High: {results['win_hi'][i] / num_sims * 100:.2f}%")
        st.write(f"Tie High: {results['tie_hi'][i] / num_sims * 100:.2f}%")
        st.write(f"Win Low:  {results['win_lo'][i] / num_sims * 100:.2f}%")
        st.write(f"Tie Low:  {results['tie_lo'][i] / num_sims * 100:.2f}%")
        st.write(f"Scoops:   {results['scoops'][i] / num_sims * 100:.2f}%")
        st.write(f"Equity:   {results['equity'][i] / num_sims * 100:.2f}%")
        st.write(f"Nut High: {results['nut_hi'][i] / num_sims * 100:.2f}%")
        st.write(f"Nut Low:  {results['nut_lo'][i] / num_sims * 100:.2f}%")

    # Chart
    st.subheader("Bar Chart")
    labels = ["Win High", "Tie High", "Win Low", "Tie Low", "Scoops", "Equity"]
    p1_vals = [results['win_hi'][0], results['tie_hi'][0], results['win_lo'][0],
               results['tie_lo'][0], results['scoops'][0], results['equity'][0] * num_sims]
    p2_vals = [results['win_hi'][1], results['tie_hi'][1], results['win_lo'][1],
               results['tie_lo'][1], results['scoops'][1], results['equity'][1] * num_sims]

    fig, ax = plt.subplots()
    x = range(len(labels))
    width = 0.35
    ax.bar([i - width/2 for i in x], [v/num_sims*100 for v in p1_vals], width, label="Player 1")
    ax.bar([i + width/2 for i in x], [v/num_sims*100 for v in p2_vals], width, label="Player 2")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("%")
    ax.legend()
    st.pyplot(fig)

    # Board example
    st.subheader("Example Runout")
    st.markdown(" ".join(suit_to_emoji(c) for c in results['example_board']), unsafe_allow_html=True)

    # CSV Export
    data = {
        "P1 Hand": [" ".join(p1_hand)],
        "P2 Hand": [" ".join(p2_hand)],
        "Board": [" ".join(board_cards)],
        "Simulations": [num_sims],
        "P1 Win Hi %": [results['win_hi'][0] / num_sims * 100],
        "P1 Tie Hi %": [results['tie_hi'][0] / num_sims * 100],
        "P1 Win Lo %": [results['win_lo'][0] / num_sims * 100],
        "P1 Tie Lo %": [results['tie_lo'][0] / num_sims * 100],
        "P1 Scoops %": [results['scoops'][0] / num_sims * 100],
        "P1 Equity %": [results['equity'][0] / num_sims * 100],
        "P1 Nut Hi %": [results['nut_hi'][0] / num_sims * 100],
        "P1 Nut Lo %": [results['nut_lo'][0] / num_sims * 100],
        "P2 Win Hi %": [results['win_hi'][1] / num_sims * 100],
        "P2 Tie Hi %": [results['tie_hi'][1] / num_sims * 100],
        "P2 Win Lo %": [results['win_lo'][1] / num_sims * 100],
        "P2 Tie Lo %": [results['tie_lo'][1] / num_sims * 100],
        "P2 Scoops %": [results['scoops'][1] / num_sims * 100],
        "P2 Equity %": [results['equity'][1] / num_sims * 100],
        "P2 Nut Hi %": [results['nut_hi'][1] / num_sims * 100],
        "P2 Nut Lo %": [results['nut_lo'][1] / num_sims * 100],
    }
    df = pd.DataFrame(data)
    st.download_button(
        "📥 Download CSV Results",
        data=df.to_csv(index=False),
        file_name=f"omaha6_sim_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )
