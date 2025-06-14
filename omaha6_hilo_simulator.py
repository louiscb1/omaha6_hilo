from card_utils import create_deck, shuffle_deck, deal_cards
from hand_evaluator import evaluate_high_hand, evaluate_low_hand
from itertools import combinations
from collections import Counter

NUM_SIMULATIONS = 10000

player1_hand = ["As", "2d", "3h", "4s", "5c", "6h"]
player2_hand = ["Ah", "Kd", "Qh", "Jc", "9d", "8s"]
partial_board = []

def qualifies_low(low_hand):
    if low_hand is None:
        return False
    return all(card[0] in "2345678A" for card in low_hand) and len(set(c[0] for c in low_hand)) == 5

def is_nut_low(low_score):
    return low_score == (5, 4, 3, 2, 1)

def run_simulation():
    win_hi = [0, 0]
    tie_hi = [0, 0]
    win_lo = [0, 0]
    tie_lo = [0, 0]
    scoops = [0, 0]
    equity = [0.0, 0.0]
    nut_hi = [0, 0]
    nut_lo = [0, 0]

    for _ in range(NUM_SIMULATIONS):
        deck = create_deck()
        known = player1_hand + player2_hand + partial_board
        for card in known:
            if card in deck:
                deck.remove(card)
        shuffle_deck(deck)

        board = partial_board.copy()
        board += deal_cards(deck, 5 - len(board))

        high_scores = []
        low_scores = []
        low_hands = []

        for hand in [player1_hand, player2_hand]:
            best_high = None
            best_low = None
            best_low_hand = None
            for hole in combinations(hand, 2):
                for board3 in combinations(board, 3):
                    full_hand = list(hole) + list(board3)
                    hi = evaluate_high_hand(full_hand)
                    lo = evaluate_low_hand(full_hand, board)
                    if best_high is None or hi > best_high:
                        best_high = hi
                    if lo is not None:
                        if best_low is None or lo < best_low:
                            best_low = lo
                            best_low_hand = full_hand
            high_scores.append(best_high)
            low_scores.append(best_low)
            low_hands.append(best_low_hand)

        max_high = max(high_scores)
        high_winners = [i for i, s in enumerate(high_scores) if s == max_high]
        for i in high_winners:
            if len(high_winners) == 1:
                win_hi[i] += 1
            else:
                tie_hi[i] += 1

        low_winners = []
        valid_lows = [s for s in low_scores if s is not None and qualifies_low(low_hands[i])]
        if valid_lows:
           min_low = min(valid_lows)
           low_winners = [i for i, s in enumerate(low_scores) if s == min_low and qualifies_low(low_hands[i])]
           for i in low_winners:
              if len(low_winners) == 1:
                 win_lo[i] += 1
           else:
                 tie_lo[i] += 1


        pot_share = [0.0, 0.0]
        for i in high_winners:
            pot_share[i] += 0.5 / len(high_winners)
        if valid_lows:
            for i in low_winners:
                pot_share[i] += 0.5 / len(low_winners)
        for i in [0, 1]:
            equity[i] += pot_share[i]

        for i in [0, 1]:
            if len(high_winners) == 1 and len(low_winners) == 1:
                if high_winners[0] == i and low_winners[0] == i:
                    scoops[i] += 1

        if high_scores[0] > high_scores[1]:
            nut_hi[0] += 1
        elif high_scores[1] > high_scores[0]:
            nut_hi[1] += 1
        if low_scores[0] and is_nut_low(low_scores[0]):
            nut_lo[0] += 1
        if low_scores[1] and is_nut_low(low_scores[1]):
            nut_lo[1] += 1

    print("--- Simulation Results ---")
    for i in [0, 1]:
        print(f"Player {i+1}:")
        print(f"  Win High %: {win_hi[i] / NUM_SIMULATIONS * 100:.2f}%")
        print(f"  Tie High %: {tie_hi[i] / NUM_SIMULATIONS * 100:.2f}%")
        print(f"  Win Low  %: {win_lo[i] / NUM_SIMULATIONS * 100:.2f}%")
        print(f"  Tie Low  %: {tie_lo[i] / NUM_SIMULATIONS * 100:.2f}%")
        print(f"  Scoop    %: {scoops[i] / NUM_SIMULATIONS * 100:.2f}%")
        print(f"  Equity   %: {equity[i] / NUM_SIMULATIONS * 100:.2f}%")
        print(f"  Nut High %: {nut_hi[i] / NUM_SIMULATIONS * 100:.2f}%")
        print(f"  Nut Low  %: {nut_lo[i] / NUM_SIMULATIONS * 100:.2f}%")

if __name__ == "__main__":
    run_simulation()
