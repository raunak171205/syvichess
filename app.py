import chess
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque
import random
import os
import threading
import time

from flask import Flask, render_template, request, jsonify

# --- ChessGame and ChessAI Classes ---
class ChessGame:
    def __init__(self):
        self.board = chess.Board()
        # Stores {'move': move, 'board_fen': board_fen_after_move, 'turn_before_move': turn_color}
        self.game_history = []
        self.ai = ChessAI()
        self.ai_eval = 0.0
        self.ai_confidence = 0.0

    def make_move(self, move_uci):
        """
        Processes a player's move and triggers the AI's response if it's Black's turn.
        
        Args:
            move_uci (str): The move in UCI format (e.g., "e2e4").
            
        Returns:
            tuple: (success_status, message, current_game_state_dict)
        """
        try:
            move = chess.Move.from_uci(move_uci)
            if move in self.board.legal_moves:
                turn_before_move = self.board.turn # Store current turn before making move
                self.board.push(move)
                self.game_history.append({'move': move, 'board_fen': self.board.fen(), 'turn_before_move': turn_before_move})
                
                # Check if game is over after player's move
                if self.board.is_game_over():
                    result = self.board.result()
                    self.ai.train_from_game(self.game_history, result)
                    self.ai.save_model() # Autosave after game ends
                    return True, "Player moved. Game Over.", self.get_game_state()
                
                # If it's AI's turn (Black) after player's move
                if self.board.turn == chess.BLACK:
                    # AI makes its move
                    ai_move, ai_confidence, ai_eval = self.ai.get_move(self.board)
                    self.ai_eval = ai_eval
                    self.ai_confidence = ai_confidence
                    
                    if ai_move:
                        turn_before_ai_move = self.board.turn
                        self.board.push(ai_move)
                        self.game_history.append({'move': ai_move, 'board_fen': self.board.fen(), 'turn_before_move': turn_before_ai_move})
                        
                        # Check if game is over after AI's move
                        if self.board.is_game_over():
                            result = self.board.result()
                            self.ai.train_from_game(self.game_history, result)
                            self.ai.save_model() # Autosave after game ends
                            return True, f"Syvi moved {ai_move.uci()}. Game Over.", self.get_game_state()
                        else:
                            return True, f"Syvi moved {ai_move.uci()}.", self.get_game_state()
                    else:
                        # This case should ideally not be reached if game is not over
                        return False, "Syvi has no legal moves (unexpected).", self.get_game_state()
                
                # If it's still White's turn (e.g., after a promotion that keeps it White's turn)
                return True, "Player moved.", self.get_game_state()

            else:
                return False, "Illegal move.", self.get_game_state()
        except ValueError:
            return False, "Invalid move format (UCI expected).", self.get_game_state()

    def reset_game(self):
        """Resets the board to the initial state and clears game history."""
        self.board = chess.Board()
        self.game_history = []
        self.ai_eval = 0.0
        self.ai_confidence = 0.0
        return True, "Game reset.", self.get_game_state()

    def undo_last_moves(self):
        """Undoes the last player's move and the AI's response, if any."""
        if len(self.game_history) >= 2:
            # Remove AI's move and player's move
            self.board.pop() 
            self.board.pop()
            self.game_history.pop() # Remove AI's history entry
            self.game_history.pop() # Remove player's history entry
            self.ai_eval = 0.0 # Reset eval after undo
            self.ai_confidence = 0.0
            return True, "Last moves undone.", self.get_game_state()
        elif len(self.game_history) == 1: # Only player's first move made
            self.board.pop() # Remove player's move
            self.game_history.pop() # Remove player's history entry
            self.ai_eval = 0.0 # Reset eval after undo
            self.ai_confidence = 0.0
            return True, "Last move undone.", self.get_game_state()
        else:
            return False, "No moves to undo.", self.get_game_state()

    def get_game_state(self):
        """Returns a dictionary representing the current state of the game."""
        return {
            "fen": self.board.fen(),
            "turn": "white" if self.board.turn == chess.WHITE else "black",
            "is_game_over": self.board.is_game_over(),
            "result": self.board.result() if self.board.is_game_over() else None,
            "is_check": self.board.is_check(),
            "syvi_eval": f"{self.ai_eval:+.2f}",
            "syvi_confidence": f"{self.ai_confidence:.0%}"
        }

class ChessAI:
    def __init__(self, model_path="chess_ai.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._create_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.model_path = model_path
        # Attempt to load the model if it exists, handle potential RuntimeError for architecture mismatch
        if os.path.exists(model_path):
            try:
                self.load_model()
            except RuntimeError as e:
                print(f"Error loading existing model: {e}. Starting with a new model. Please delete '{model_path}' if this persists.")
        self.memory = deque(maxlen=10000) # Experience replay memory
        self.gamma = 0.99 # Discount factor for rewards
        self.temperature = 1.0 # Softmax temperature for exploration

    def _create_model(self):
        """
        Creates the neural network model for the Chess AI.
        Input: 8x8x12 channels (piece type and color for each square).
        Output: 4096 (64 'from' squares * 64 'to' squares) possible move indices.
        """
        model = nn.Sequential(
            nn.Linear(8*8*12, 256), # Input layer to hidden layer
            nn.ReLU(),              # Activation function
            nn.Linear(256, 256),    # Second hidden layer
            nn.ReLU(),              # Activation function
            nn.Linear(256, 4096)    # Output layer (predicts scores for each possible move)
        )
        return model.to(self.device)

    def board_to_tensor(self, board):
        """
        Converts a python-chess board object into a 1D tensor representation
        suitable for the neural network.
        Each square has 12 channels: 6 for white pieces (P, N, B, R, Q, K)
        and 6 for black pieces (p, n, b, r, q, k).
        """
        tensor = torch.zeros(8*8*12, dtype=torch.float32)
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                # Map piece type and color to a specific channel
                # Channels 0-5 for white pieces, 6-11 for black pieces
                channel = piece.piece_type - 1 + (6 if piece.color == chess.BLACK else 0)
                tensor[channel * 64 + square] = 1 # Set the corresponding feature to 1
        return tensor.to(self.device)

    def get_move(self, board):
        """
        Selects a move for the AI based on the current board state and the trained model.
        
        Args:
            board (chess.Board): The current chess board state.
            
        Returns:
            tuple: (selected_move, confidence, evaluation)
                   selected_move (chess.Move or None): The chosen move.
                   confidence (float): Probability of the chosen move.
                   evaluation (float): Simple evaluation based on confidence.
        """
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None, 0, 0 # No legal moves, game is likely over

        state = self.board_to_tensor(board)
        with torch.no_grad(): # Disable gradient calculation for inference
            logits = self.model(state.unsqueeze(0)) # Add batch dimension

        # Filter logits to only consider legal moves
        legal_move_indices = [self.move_to_index(move) for move in legal_moves]
        
        # Create a mask: -inf for illegal moves, 0 for legal moves
        mask = torch.full_like(logits[0], float('-inf'))
        mask[legal_move_indices] = 0
        
        # Apply mask to logits to ensure only legal moves have finite probabilities
        masked_logits = logits[0] + mask
        
        # Apply softmax to get probabilities over legal moves
        move_probs = F.softmax(masked_logits / self.temperature, dim=0)

        # Handle the case where all legal moves have extremely low probabilities (sum is close to zero)
        if move_probs.sum().item() < 1e-6:
            # Fallback: choose a random legal move if probabilities are too low
            selected_move = random.choice(legal_moves)
            confidence = 1.0 / len(legal_moves) # Equal probability for all legal moves
            evaluation = 0.0 # Cannot reliably evaluate in this case
            return selected_move, confidence, evaluation

        # Sample a move based on the probabilities
        move_idx = torch.multinomial(move_probs, 1).item()
        selected_move = self.index_to_move(move_idx)
        confidence = move_probs[move_idx].item()
        evaluation = confidence * 2 - 1 # Simple evaluation: higher confidence -> better eval

        # Final check: ensure the selected move is actually legal (should be due to masking)
        if selected_move not in legal_moves:
            # Fallback: if an illegal move was somehow selected, pick the legal move
            # with the highest probability. This is a robust fallback.
            max_prob = -1
            best_legal_move = None
            for move in legal_moves:
                idx = self.move_to_index(move)
                if move_probs[idx].item() > max_prob:
                    max_prob = move_probs[idx].item()
                    best_legal_move = move
            selected_move = best_legal_move
            confidence = max_prob
            evaluation = confidence * 2 - 1

        return selected_move, confidence, evaluation

    def train_from_game(self, game_history, result):
        """
        Trains the AI model based on the outcome of a completed game.
        
        Args:
            game_history (list): List of dictionaries, each representing a move transition.
            result (str): Game result string (e.g., "1-0", "0-1", "1/2-1/2").
        """
        # Map game result to a numerical reward for the AI (Black)
        # AI wins (0-1) -> positive reward (1)
        # AI loses (1-0) -> negative reward (-1)
        # Draw (1/2-1/2) -> neutral reward (0)
        result_value = {'1-0': -1, '0-1': 1, '1/2-1/2': 0}.get(result, 0)
        
        # Filter game history to only include AI's (Black's) turns
        ai_transitions = [t for t in game_history if t['turn_before_move'] == chess.BLACK]

        for i, transition in enumerate(ai_transitions):
            # Reconstruct the board state at the point the AI made its move
            board_at_ai_turn = chess.Board(transition['board_fen'])
            state = self.board_to_tensor(board_at_ai_turn)
            
            move = transition['move']
            # Calculate discounted reward: rewards for earlier moves are less impactful
            reward = result_value * (self.gamma ** (len(ai_transitions) - i - 1))
            self.memory.append((state, self.move_to_index(move), reward))

        # Train the model if enough samples are in memory
        if len(self.memory) >= 128:
            batch = random.sample(self.memory, 128) # Sample a batch from memory
            states = torch.stack([x[0] for x in batch])
            moves = torch.tensor([x[1] for x in batch], device=self.device)
            rewards = torch.tensor([x[2] for x in batch], device=self.device)

            logits = self.model(states)
            
            # Compute log probabilities for the selected moves
            log_probs = F.log_softmax(logits, dim=1)
            
            # Gather the log probabilities corresponding to the actual moves taken
            selected_log_probs = log_probs.gather(1, moves.unsqueeze(-1)).squeeze(-1)
            
            # Compute loss using policy gradient-like approach: - (reward * log_prob)
            # Minimizing this loss means increasing log_prob for positive rewards
            # and decreasing log_prob for negative rewards.
            loss = - (rewards * selected_log_probs).mean()
            
            self.optimizer.zero_grad() # Clear previous gradients
            loss.backward()             # Backpropagate the loss
            self.optimizer.step()      # Update model weights

    def save_model(self):
        """Saves the current state of the AI model and optimizer."""
        torch.save({
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
        }, self.model_path)
        print(f"AI knowledge saved to {self.model_path}")

    def load_model(self):
        """Loads the AI model and optimizer state from a file."""
        if os.path.exists(self.model_path):
            checkpoint = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
            print(f"Loaded AI knowledge from {self.model_path}")

    def move_to_index(self, move):
        """Converts a chess.Move object to a unique integer index (0-4095)."""
        return move.from_square * 64 + move.to_square

    def index_to_move(self, idx):
        """Converts an integer index back to a chess.Move object."""
        from_sq = idx // 64
        to_sq = idx % 64
        # Note: This simple conversion doesn't handle promotions explicitly.
        # For a full chess engine, promotions would require more complex indexing.
        return chess.Move(from_sq, to_sq)

# --- Flask App Setup ---
app = Flask(__name__)

# Initialize the game instance globally
# For a multi-user application, you would need to manage separate game instances per user (e.g., using Flask sessions or a database)
current_game = ChessGame()

# Autosave mechanism (runs in a separate thread)
AUTOSAVE_INTERVAL_SECONDS = 300 # Autosave every 5 minutes

def autosave_loop():
    """Background thread function to periodically save the AI model."""
    while True:
        time.sleep(AUTOSAVE_INTERVAL_SECONDS)
        print("Autosaving AI knowledge...")
        current_game.ai.save_model()
        print("Autosave complete.")

# Start the autosave thread as a daemon so it exits when the main program exits
autosave_thread = threading.Thread(target=autosave_loop, daemon=True)
autosave_thread.start()

@app.route('/')
def index():
    """Serves the main HTML page for the chess game."""
    # Flask will automatically look for 'index.html' in the 'templates' folder
    return render_template('index.html')

@app.route('/api/start_game', methods=['POST'])
def start_game_api():
    """API endpoint to reset the game and return the initial state."""
    success, message, game_state = current_game.reset_game()
    return jsonify({"success": success, "message": message, "game_state": game_state})

@app.route('/api/make_move', methods=['POST'])
def make_move_api():
    """API endpoint to receive a player's move, make it, and get the AI's response."""
    data = request.get_json()
    move_uci = data.get('move_uci')

    if not move_uci:
        return jsonify({"success": False, "message": "No move_uci provided."}), 400

    success, message, game_state = current_game.make_move(move_uci)
    return jsonify({"success": success, "message": message, "game_state": game_state})

@app.route('/api/undo', methods=['POST'])
def undo_move_api():
    """API endpoint to undo the last player and AI moves."""
    success, message, game_state = current_game.undo_last_moves()
    return jsonify({"success": success, "message": message, "game_state": game_state})

@app.route('/api/get_state', methods=['GET'])
def get_state_api():
    """API endpoint to return the current game state."""
    game_state = current_game.get_game_state()
    return jsonify({"success": True, "game_state": game_state})

if __name__ == '__main__':
    # Ensure the 'templates' folder exists for Flask to find index.html
    os.makedirs('templates', exist_ok=True)
    
    # Run the Flask app
    # debug=True enables reloader and debugger, useful for development
    # host='0.0.0.0' makes the server accessible from other devices on the network
    app.run(debug=True, host='0.0.0.0', port=5000)

