import random
from card_utils import create_deck, shuffle_deck, deal_cards
from hand_evaluator import evaluate_high_hand, evaluate_low_hand
from itertools import combinations
from tqdm import tqdm  # Progress bar

# Config
NUM_PLAYERS = 2
NUM_SIMULATIONS = 100_000  # You can scale this to 1_000_000 later

# You can input specific player hands here (leave as None for random)
# Example: PLAYER_HANDS = ["Ah2s3c4d5h6h", "AcAdAsAhKdKh"]
PLAYER_HANDS = ["Ah2s3c4d5h6h", None]

# You can input partial board here (5 cards max). Example: BOARD_CARDS = ['Ah', 'Kd', 'Qc']
BOARD_CARDS = []

# Initialize win counters
high_wins = [0 for _ in range(NUM_PLAYERS)]
low_wins = [0 for _ in range(NUM_PLAYERS)]
scoops = [0 for _ in range(NUM_PLAYERS)]

# Helper to parse hand string like "Ah2s3c4d5h6h"
def parse_hand_string(hand_str):
    valid_ranks = "23456789TJQKA"
    valid_suits = "cdhs"

    # Split string into list of 2-char card codes
    cards = [hand_str[i:i+2] for i in range(0, len(hand_str), 2)]

    if len(cards) != 6:
        raise ValueError("Each Omaha 6 hand must have exactly 6 cards (12 characters).")

    # Check each card is valid
    for card in cards:
        if len(card) != 2 or card[0] not in valid_ranks or card[1] not in valid_suits:
            raise ValueError(f"Invalid card format: '{card}'")

    return cards

# Sanity checker to catch duplicate cards
def check_for_duplicate_cards(all_player_hands, board_cards):
    seen = set()
    for hand in all_player_hands:
        for card in hand:
            if card in seen:
                raise ValueError(f"Duplicate card detected: {card}")
            seen.add(card)
    for card in board_cards:
        if card in seen:
            raise ValueError(f"Duplicate card detected between player hands and board: {card}")
        seen.add(card)

# Main simulation loop
for sim in tqdm(range(NUM_SIMULATIONS)):
    deck = create_deck()
    shuffle_deck(deck)

    # Deal player hands
    player_hands = []
    for i in range(NUM_PLAYERS):
        if PLAYER_HANDS[i] is not None:
            hand = parse_hand_string(PLAYER_HANDS[i])
            # Remove specified cards from deck
            for card in hand:
                deck.remove(card)
            player_hands.append(hand)
        else:
            player_hand = deal_cards(deck, 6)
            player_hands.append(player_hand)

    # Copy board input
    board = BOARD_CARDS.copy()
    # Remove board cards from deck
    for card in board:
        deck.remove(card)
    # Deal remaining board cards
    num_to_deal = 5 - len(board)
    if num_to_deal > 0:
        board += deal_cards(deck, num_to_deal)

    # Run sanity check once on first sim
    if sim == 0:
        check_for_duplicate_cards(player_hands, board)

    # For each player, find best High and Low hand (2 hole + 3 board rule)
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

    # Determine Low winner (if any Low)
    valid_lows = [score for score in low_scores if score is not None]
    min_low = None  # Initialize
    if valid_lows:
        min_low = min(valid_lows)
        low_winners = [i for i, score in enumerate(low_scores) if score == min_low]
        for w in low_winners:
            low_wins[w] += 1 / len(low_winners)

    # Determine scoops (player who wins both High and Low alone)
    for i in range(NUM_PLAYERS):
        if min_low is not None and \
           high_scores[i] == max_high and low_scores[i] == min_low and \
           high_scores.count(max_high) == 1 and low_scores.count(min_low) == 1:
            scoops[i] += 1

# Print results
print(f"\nResults after {NUM_SIMULATIONS:,} simulations:")
for i in range(NUM_PLAYERS):
    print(f"\nPlayer {i+1}:")
    print(f"  High wins: {100 * high_wins[i] / NUM_SIMULATIONS:.2f}%")
    print(f"  Low wins: {100 * low_wins[i] / NUM_SIMULATIONS:.2f}%")
    print(f"  Scoops:    {100 * scoops[i] / NUM_SIMULATIONS:.2f}%")
