"""
Density Chess — v1.0.0
======================

Licence
-------
The whole project is under GNU General Public Licence v3.0+.

Description
-----------
Density Chess analyzes a chessboard and shows the density repartition on it.
It also could play a game and show the evolution of the density in an animation.

About *.chess files
-------------------
*.chess files allow to save a chessboard, load one and make moves.

A chess file of a chess board is always formated as follows:
 – the first char is the color of the piece ('w' or 'b'');
 – the two seconds are the piece itself ('Qu', 'Ki', 'Ro', 'Bi', 'Kn', 'Pa');
 – if the square of the chessboard is empty the code is '...'.

e.g.: an exemple of a new game chess file
```
bRo bKn bBi bQu bKi bBi bKn bRo
bPa bPa bPa bPa bPa bPa bPa bPa
... ... ... ... ... ... ... ...
... ... ... ... ... ... ... ...
... ... ... ... ... ... ... ...
... ... ... ... ... ... ... ...
wPa wPa wPa wPa wPa wPa wPa wPa
wRo wKn wBi wQu wKi wBi wKn wRo
```

A chess file of moves is just composed of the moves in algebric code with one move per line.
e.g.:
```
e2-e4
g8-f6
f1-b5
b8-c6
g1-f3
e7-e5
```

Exemple of usage
----------------
Assuming Density Chess was imported as follows:

>>> import density_chess as dc
>>> chess = dc.ChessboardDensity()

To start a new game :

>>> chess.new_game()

To see the chessboard

>>> chess.get_fig() # it should return an empty list

To make an animation with some moves

>>> chess.get_animation("moves", "chess_game", interval=500)

Where "moves" is the file which contains the moves, and "chess_game" the GIF's name.

You can also save and load games:

>>> chess.save_game("chess_game")
>>> chess.load_game("chess_game")
"""
import matplotlib.pyplot as plt
import numpy as np

from matplotlib import animation
from matplotlib.patches import Rectangle


# ┌─────────┐ #
# │ Classes │ #
# └─────────┘ #
class ChessboardDensity:
    """Modelling a chessboard, shows the density repartition and its evolution during a game.

    Attributes
    ----------
    density_table : np.array
        Table of density.
    pieces_table : np.array
        Table of pieces.

    Methods
    -------
    new_game
        Start a new game and place the pieces.
    load_game
        Load an existing game from a *.chess file.
    save_game
        Save the current game to a *.chess file.
    get_fig
        Show the chess table with the density.
    get_animation
        Show the evolution of the density during a game.
    move
        Make a move with algebric code.
    """
    def __init__(self):
        """Initialize the instances."""
        self.density_table = np.array([[0 for _ in range(8)] for _ in range(8)])
        self.pieces_table = np.array([[(-1, -1) for _ in range(8)] for _ in range(8)])

    # Private methods
    def __check_diagonal(self, pos_x: int, pos_y: int, piece_value: int, isblack: bool):
        """Compute the zone of influence on diagonal lines

        Parameters
        ----------
        pos_x : int
            Position on x-axis.
        pos_y : int
            Position on y-axis.
        piece_value : int
            Value of the current piece.
        isblack : bool
            The piece's color.
        """
        filled_directions = []
        for offset in range(1, 8):
            if 0 not in filled_directions and check_case(pos_x + offset, pos_y + offset):
                if self.pieces_table[pos_y + offset, pos_x + offset][0] != -1:
                    filled_directions.append(0)
                self.density_table[pos_y + offset, pos_x + offset] += (piece_value *
                        (-1) ** isblack)

            if 1 not in filled_directions and check_case(pos_x + offset, pos_y - offset):
                if self.pieces_table[pos_y - offset, pos_x + offset][0] != -1:
                    filled_directions.append(1)
                self.density_table[pos_y - offset, pos_x + offset] += (piece_value *
                        (-1) ** isblack)

            if 2 not in filled_directions and check_case(pos_x - offset, pos_y + offset):
                if self.pieces_table[pos_y + offset, pos_x - offset][0] != -1:
                    filled_directions.append(2)
                self.density_table[pos_y + offset, pos_x - offset] += (piece_value *
                        (-1) ** isblack)

            if 3 not in filled_directions and check_case(pos_x - offset, pos_y - offset):
                if self.pieces_table[pos_y - offset, pos_x - offset][0] != -1:
                    filled_directions.append(3)
                self.density_table[pos_y - offset, pos_x - offset] += (piece_value *
                        (-1) ** isblack)

    def __check_straight(self, pos_x: int, pos_y: int, piece_value: int, isblack: bool):
        """Compute the zone of influence on straight lines

        Parameters
        ----------
        pos_x : int
            Position on x-axis.
        pos_y : int
            Position on y-axis.
        piece_value : int
            Value of the current piece.
        isblack : bool
            The piece's color.
        """
        for x_offset in range(8):
            if x_offset and check_case(pos_x + x_offset, pos_y):
                self.density_table[pos_y, pos_x + x_offset] += piece_value * (-1) ** isblack
                if self.pieces_table[pos_y, pos_x + x_offset][0] != -1:
                    break
        for x_offset in range(8):
            if x_offset and check_case(pos_x - x_offset, pos_y):
                self.density_table[pos_y, pos_x - x_offset] += piece_value * (-1) ** isblack
                if self.pieces_table[pos_y, pos_x - x_offset][0] != -1:
                    break
        for y_offset in range(8):
            if y_offset and check_case(pos_x, pos_y + y_offset):
                self.density_table[pos_y + y_offset, pos_x] += piece_value * (-1) ** isblack
                if self.pieces_table[pos_y + y_offset, pos_x][0] != -1:
                    break
        for y_offset in range(8):
            if y_offset and check_case(pos_x, pos_y - y_offset):
                self.density_table[pos_y - y_offset, pos_x] += piece_value * (-1) ** isblack
                if self.pieces_table[pos_y - y_offset, pos_x][0] != -1:
                    break

    def __queen(self, pos_x: int, pos_y: int, isblack: bool):
        """Compute the queen's zone of influence."""
        self.density_table[pos_y, pos_x] += QUEEN_VALUE * (-1) ** isblack

        self.__check_diagonal(pos_x, pos_y, QUEEN_VALUE, isblack)
        self.__check_straight(pos_x, pos_y, QUEEN_VALUE, isblack)

    def __king(self, pos_x: int, pos_y: int, isblack: bool):
        """Compute the king's zone of influence."""
        self.density_table[pos_y, pos_x] += KING_VALUE * (-1) ** isblack

        for x_offset in range(-1, 2):
            for y_offset in range(-1, 2):
                if ((x_offset or y_offset)
                        and check_case(pos_x + x_offset, pos_y + y_offset)):
                    self.density_table[pos_y + y_offset, pos_x + x_offset] += (KING_VALUE *
                            (-1) ** isblack)

    def __rook(self, pos_x: int, pos_y: int, isblack: bool):
        """Compute the rook's zone of influence."""
        self.density_table[pos_y, pos_x] += ROOK_VALUE * (-1) ** isblack

        self.__check_straight(pos_x, pos_y, ROOK_VALUE, isblack)

    def __bishop(self, pos_x: int, pos_y: int, isblack: bool):
        """Compute the bishop's zone of influence."""
        self.density_table[pos_y, pos_x] += BISHOP_VALUE * (-1) ** isblack
        self.__check_diagonal(pos_x, pos_y, BISHOP_VALUE, isblack)

    def __knight(self, pos_x: int, pos_y: int, isblack: bool):
        """Compute the knight's zone of influence."""
        self.density_table[pos_y, pos_x] += (KNIGHT_VALUE * (-1) ** isblack)

        positions = [
                (pos_x - 2, pos_y + 1),
                (pos_x - 2, pos_y - 1),
                (pos_x + 2, pos_y + 1),
                (pos_x + 2, pos_y - 1),
                (pos_x + 1, pos_y + 2),
                (pos_x - 1, pos_y + 2),
                (pos_x + 1, pos_y - 2),
                (pos_x - 1, pos_y - 2)
            ]
        for position_x, position_y in positions:
            if check_case(position_x, position_y):
                self.density_table[position_y, position_x] += (KNIGHT_VALUE * (-1) ** isblack)

    def __pawn(self, pos_x: int, pos_y: int, isblack: bool):
        """Compute the pawn's zone of influence."""
        self.density_table[pos_y, pos_x] += PAWN_VALUE * (-1) ** isblack

        if check_case(pos_x + 1, pos_y + (-1) ** isblack):
            self.density_table[pos_y + (-1) ** isblack, pos_x + 1] += PAWN_VALUE * (-1) ** isblack
        if check_case(pos_x - 1, pos_y + (-1) ** isblack):
            self.density_table[pos_y + (-1) ** isblack, pos_x - 1] += PAWN_VALUE * (-1) ** isblack

    def __compute_density(self):
        """Compute the density of the chess table."""
        pieces_functions = [
                self.__queen,
                self.__king,
                self.__rook,
                self.__bishop,
                self.__knight,
                self.__pawn
            ]
        self.density_table = np.array([[0 for _ in range(8)] for _ in range(8)])

        for targeted_x in range(8):
            for targeted_y in range(8):
                piece_index, piece_isblack = self.pieces_table[targeted_y, targeted_x]
                if piece_index != -1:
                    pieces_functions[piece_index](targeted_x, targeted_y, piece_isblack)

    # Public methods
    def new_game(self):
        """Place the pieces for a new game."""
        self.pieces_table = np.array([[(-1, -1) for _ in range(8)] for _ in range(8)])

        self.pieces_table[0, 0] = (2, False)
        self.pieces_table[0, 7] = (2, False)
        self.pieces_table[0, 1] = (4, False)
        self.pieces_table[0, 6] = (4, False)
        self.pieces_table[0, 2] = (3, False)
        self.pieces_table[0, 5] = (3, False)
        self.pieces_table[0, 3] = (0, False)
        self.pieces_table[0, 4] = (1, False)

        self.pieces_table[7, 0] = (2, True)
        self.pieces_table[7, 7] = (2, True)
        self.pieces_table[7, 1] = (4, True)
        self.pieces_table[7, 6] = (4, True)
        self.pieces_table[7, 2] = (3, True)
        self.pieces_table[7, 5] = (3, True)
        self.pieces_table[7, 3] = (0, True)
        self.pieces_table[7, 4] = (1, True)

        for i in range(8):
            self.pieces_table[1, i] = (5, False)
            self.pieces_table[6, i] = (5, True)

    def load_game(self, filename: str):
        """Load a *.chess file.

        Parameters
        ----------
        filename : str
            The name of the file (without extension) to be read.
        """
        with open(f"{filename}.chess", "r", encoding="utf-8") as chess_file:
            lines = [line.rstrip() for line in chess_file.readlines()]

        lines.reverse()
        for line, line_content in enumerate(lines):
            for column, piece in enumerate(line_content.split(" ")):
                isblack = PIECES_COLORS.index(piece[0].lower())
                piece_id = PIECES_SYMBOLS.index(piece[1: ].lower())

                self.add_piece(column, line, piece_id - 1, isblack - 1)

        self.__compute_density()

    def save_game(self, filename: str):
        """Save the game into a *.chess file.

        Parameters
        ----------
        filename : str
            The name of the file (without extension) in which the game should be saved.
        """
        lines = []
        for targeted_y in range(8):
            line = " ".join(
                    [PIECES_COLORS[color + 1] + PIECES_SYMBOLS[index + 1].capitalize()
                    for index, color in self.pieces_table[targeted_y, :]]
                )
            lines.append(line)

        lines.reverse()
        with open(f"{filename}.chess", "w", encoding="utf-8") as chess_file:
            chess_file.write("\n".join(lines))

    def get_fig(self, axes: plt.Axes=None, display: bool=True):
        """Return the chess table. In red kinds, the white pieces, in blue kinds the black ones.

        Parameters
        ----------
        axes : plt.Axes, optionnal
            Axes on wich the function will draw.
            By default, new axes is created.
        display : bool
            If the figure should be shown.
        Returns
        -------
        data : list
            List of matplotlib objects to compute an animation.
        """
        self.__compute_density()
        if not axes:
            plt.figure(figsize=(7, 7))
            axes = plt.axes()
        data = []
        pieces = (("♕", "♔", "♖", "♗", "♘", "♙"), ("♛", "♚", "♜", "♝", "♞", "♟"))

        for targeted_x in range(8):
            for targeted_y in range(8):
                # Density
                color = get_color(
                        self.density_table[targeted_y, targeted_x],
                        np.max(np.abs(self.density_table))
                    )
                data.append(axes.add_patch(Rectangle(
                        (targeted_x / 8, targeted_y / 8),
                        1/8,
                        1/8,
                        color=color
                    )))

                # Piece symbol
                piece_index, piece_isblack = self.pieces_table[targeted_y, targeted_x]
                if piece_index != -1:
                    data.append(plt.text(
                            targeted_x / 8 + 1/32,
                            targeted_y / 8 + 1/32,
                            pieces[piece_isblack][piece_index],
                            fontsize=20
                        ))

        # Grid and axes
        for index in np.arange(0, 1 + 1/8, 1/8):
            plt.plot((index, index), (0, 1), color=(0.6, 0.6, 0.6), linewidth=1)
            plt.plot((0, 1), (index, index), color=(0.6, 0.6, 0.6), linewidth=1)

        plt.xticks((np.arange(1/16, 1, 1/8)), ("a", "b", "c", "d", "e", "f", "g", "h"))
        plt.yticks((np.arange(1/16, 1, 1/8)), (1, 2, 3, 4, 5, 6, 7, 8))

        plt.xlim((0, 1))
        plt.ylim((0, 1))

        if display:
            plt.show()
            return []

        return data

    def get_animation(self, intput_filename: str, output_filename: str, interval: int=500):
        """Make the game evolves with the moves givens.

        Parameters
        ----------
        intput_filename : str
            The name of the file wich contains algebric codes for moves.
        output_filename : str
            The name of the GIF file.
        interval : int
            Time between two frames of the animation in ms.
    
        Exemples
        --------
        Assuming `chess` is a ChessboardDensity instance:

        >>> chess.new_game() # start a new_game
        >>> chess.get_animation("moves", chess_game", interval=500)
        """
        fig = plt.figure(figsize=(7, 7))
        axes = plt.axes()
        images = [self.get_fig(axes, False)]

        with open(f"{intput_filename}.chess", "r", encoding="utf-8") as chess_file:
            moves = [move.rstrip() for move in chess_file.readlines()]

        for move in moves:
            self.move(move)
            images.append(self.get_fig(axes, False))

        chess_ani = animation.ArtistAnimation(fig, images, interval=interval)
        chess_ani.save(output_filename + ".gif")

    def move(self, algebric_move: str):
        """Allow to move a piece with long algebric codes.

        Parameters
        ----------
        algebric_move : str
            The algebric code for the move.

        Exemple
        -------
        Assuming `chess` is a ChessboardDensity instance:

        >>> chess.new_game() # start a new game
        >>> chess.move("e2-e4") # move the pawn on e4 to e5
        """

        columns = ("a", "b", "c", "d", "e", "f", "g", "h")
        algebric_move = algebric_move.lower()

        if algebric_move[0] not in columns or algebric_move[3] not in columns:
            raise ChessMoveError("invalid column, expected column between 'a' and 'h'")
        if not algebric_move[1].isdigit() or not algebric_move[4].isdigit():
            raise ChessMoveError("invalid line, expected a number between 1 and 8")

        column_start = columns.index(algebric_move[0])
        line_start = int(algebric_move[1]) - 1
        column_end = columns.index(algebric_move[3])
        line_end = int(algebric_move[4]) - 1

        if not 0 <= line_start < 8 or not 0 <= line_end < 8:
            raise ChessMoveError("invalid line, expected a number between 1 and 8 but")

        self.pieces_table[line_end, column_end] = self.pieces_table[line_start, column_start]
        self.pieces_table[line_start, column_start] = (-1, -1)

    def add_piece(self, pos_x: int, pos_y: int, piece: int, isblack: bool):
        """Add a piece to the chessboard.

        Parameters
        ----------
        pos_x : int
            Position on the x-axis of the new piece.
        pos_y : int
            Position on the y-axis of the new piece.
        piece : int
            The id of the piece.
                0: Queen
                1: King
                2: Rook
                3: Bishop
                4: Knight
                5: Pawn
        isblack : bool
            If ``isblack=False`` the added piece is white.
            If ``isblack=True`` the added piece is black.
        """
        self.pieces_table[pos_y, pos_x] = (piece, isblack)


class ChessMoveError(Exception):
    """Error in algebric notation."""


# ┌───────────┐ #
# │ Functions │ #
# └───────────┘ #
def check_case(pos_x: int, pos_y: int):
    """Check if the case is in the chess table.

    Parameters
    ----------
    pos_x : int
        Position on x-axis.
        Between 0 and 7.
    pos_y : int
        Position on y-axis.
        Between 0 and 7.

    Returns
    -------
    out : bool
        Returns `True` if the case is in the chess table, `False` else.
    """

    if (0 <= pos_x < 8) and (0 <= pos_y < 8):
        return True
    return False


def get_color(value: int, max_value: int):
    """Compute the color of the value with the maximum density value.

    Parameters
    ----------
    value : int
        Density of the looked case.
    max_value : int
        Maximum density value of the chessboard.

    Returns
    -------
    out : tuple
        RGB tuple encoding the color.
    """
    value += max_value
    color = value / max_value
    if value <= max_value:
        return (color, color, 1)
    return (1, 2 - color, 2 - color)

def move_file(filename: str, *moves):
    """Create a *.chess file with the moves given.

    Parameters
    ----------
    filename : str
        Name of the output file (without extension).
    moves : tuple
        Moves in algebric code

    Exemple
    -------
    To make a file with somes moves:
    >>> move_file("moves", "e2-e4", "e7-e5")
    """
    with open(f"{filename}.chess", "w", encoding="utf-8") as chess_file:
        chess_file.write("\n".join(moves))


# ┌──────┐ #
# │ Data │ #
# └──────┘ #
QUEEN_VALUE = 2
KING_VALUE = 1
ROOK_VALUE = 5
BISHOP_VALUE = 3
KNIGHT_VALUE = 3
PAWN_VALUE = 9

PIECES_SYMBOLS = ("..", "qu", "ki", "ro", "bi", "kn", "pa")
PIECES_COLORS = (".", "w", "b")
