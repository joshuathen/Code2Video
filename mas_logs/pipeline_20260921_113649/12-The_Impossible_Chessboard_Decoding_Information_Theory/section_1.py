from manim import *

class TeachingScene(Scene):
    def setup_layout(self, title_text, lecture_lines):
        # BASE
        self.camera.background_color = "#000000"
        self.title = Text(title_text, font_size=28, color=WHITE).to_edge(UP)
        self.add(self.title)

        # Left-side lecture content (bullets with "-")
        lecture_texts = [Text(line, font_size=22, color=WHITE) for line in lecture_lines]
        self.lecture = VGroup(*lecture_texts).arrange(DOWN, aligned_edge=LEFT).scale(0.8)
        self.lecture.to_edge(LEFT, buff=0.2)
        self.add(self.lecture)

        # Define fine-grained animation grid (4x4 grid on right side)
        self.grid = {}
        rows = ["A", "B", "C", "D", "E", "F"]  # Top to bottom
        cols = ["1", "2", "3", "4", "5", "6"]  # Left to right

        for i, row in enumerate(rows):
            for j, col in enumerate(cols):
                x = 0.5 + j * 1
                y = 2.2 - i * 1
                self.grid[f"{row}{col}"] = np.array([x, y, 0])

    def place_at_grid(self, mobject, grid_pos, scale_factor=1.0):
        mobject.scale(scale_factor)
        mobject.move_to(self.grid[grid_pos])
        return mobject

    def place_in_area(self, mobject, top_left, bottom_right, scale_factor=1.0):
        tl_pos = self.grid[top_left]
        br_pos = self.grid[bottom_right]
        
        # Calculate center of the area
        center_x = (tl_pos[0] + br_pos[0]) / 2
        center_y = (tl_pos[1] + br_pos[1]) / 2
        center = np.array([center_x, center_y, 0])
        
        mobject.scale(scale_factor)
        mobject.move_to(center)
        return mobject

class Section1Scene(TeachingScene):
    def construct(self):
        self.setup_layout("The Hook: The Prisoner's Dilemma", [
            "Sixty-four coins sit on a chessboard.",
            "One coin is chosen as the target.",
            "Alice sees all coins and selects one.",
            "Can she help Bob guess correctly?",
            "It’s a classic problem of information."
        ])
        
        # Create Chessboard from asset
        board = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/chessboard.svg")
        self.place_in_area(board, 'C2', 'F6', scale_factor=0.8)

        # Assets
        alice = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/alice.svg")
        bob = Text("Bob", font_size=24) # Placeholder for missing SVG
        coin = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/coin.svg")

        # Labels
        alice_label = Text("Alice", font_size=16)
        bob_label = Text("Bob", font_size=16)

        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(board))
        self.lecture[0].set_color(YELLOW)

        # === Animation for Lecture Line 2 ===
        target_indicator = coin.copy()
        self.place_at_grid(target_indicator, 'D4', scale_factor=0.5)
        self.play(Create(target_indicator))
        self.lecture[1].set_color(YELLOW)

        # === Animation for Lecture Line 3 ===
        self.place_at_grid(alice, 'B3', scale_factor=0.6)
        self.place_at_grid(bob, 'B5', scale_factor=0.6)
        alice_label.next_to(alice, UP, buff=0.1)
        bob_label.next_to(bob, UP, buff=0.1)
        self.play(FadeIn(alice), FadeIn(alice_label), FadeIn(bob), FadeIn(bob_label))
        self.lecture[2].set_color(YELLOW)

        # === Animation for Lecture Line 4 ===
        question_mark = Text("?", font_size=40, color=WHITE).next_to(bob, RIGHT)
        self.play(Write(question_mark))
        self.lecture[3].set_color(YELLOW)

        # === Animation for Lecture Line 5 ===
        self.play(Flash(board, color=YELLOW, line_length=0.1, num_lines=10))
        self.lecture[4].set_color(YELLOW)
        self.wait(2)
