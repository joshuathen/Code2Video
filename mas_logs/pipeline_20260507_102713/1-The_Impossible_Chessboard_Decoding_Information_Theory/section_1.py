from manim import *
import random

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
        # Initial Scene Setup
        title = "The Warden's Impossible Game"
        lines = [
            'Meet Alice and Bob, two prisoners facing a challenge.',
            'A chessboard with sixty-four coins, randomly heads or tails.',
            'Alice flips one coin to signal a secret target.'
        ]
        self.setup_layout(title, lines)

        # === Animation for Lecture Line 1 ===
        # Alice (cat icon placeholder) and Bob (dog icon placeholder)
        # Using styled icons with colors #FFD700 and #1E90FF
        alice = VGroup(
            Circle(radius=0.4, color="#FFD700", fill_opacity=0.2),
            Text("Alice", font_size=24, color="#FFD700")
        ).arrange(DOWN, buff=0.1)
        
        bob = VGroup(
            Circle(radius=0.4, color="#1E90FF", fill_opacity=0.2),
            Text("Bob", font_size=24, color="#1E90FF")
        ).arrange(DOWN, buff=0.1)

        self.place_at_grid(alice, "A1", scale_factor=0.8)
        self.place_at_grid(bob, "A6", scale_factor=0.8)

        self.play(
            FadeIn(alice),
            FadeIn(bob),
            self.lecture[0].animate.set_color("#FFD700")
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Creating an 8x8 grid of coins
        random.seed(42)  # For consistent state generation
        board_group = VGroup()
        coin_states = []
        
        for r in range(8):
            row_vgroup = VGroup()
            for c in range(8):
                state = random.choice(["H", "T"])
                coin_states.append(state)
                
                circle = Circle(radius=0.18, stroke_width=2, color=WHITE)
                label = Text(state, font_size=12, color=WHITE if state == "H" else "#888888")
                
                coin_unit = VGroup(circle, label)
                coin_unit.move_to([c * 0.42, -r * 0.42, 0])
                row_vgroup.add(coin_unit)
            board_group.add(row_vgroup)

        self.place_in_area(board_group, "B1", "F6", scale_factor=0.85)

        self.play(
            Create(board_group),
            self.lecture[1].animate.set_color("#FFFFFF")
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Highlight Target Square (Square #42)
        # 8x8 index 42 -> Row 5, Col 2 (0-indexed)
        target_row, target_col = 5, 2
        target_coin_unit = board_group[target_row][target_col]
        target_highlight = SurroundingRectangle(target_coin_unit, color="#FF0000", buff=0.05)

        # Highlight Alice's flip (Square #10 -> Row 1, Col 2)
        flip_row, flip_col = 1, 2
        flip_coin_unit = board_group[flip_row][flip_col]
        flip_circle = flip_coin_unit[0]
        flip_label = flip_coin_unit[1]

        self.play(
            Create(target_highlight),
            self.lecture[2].animate.set_color("#00FF00")
        )
        self.wait(0.5)

        # Visualizing the coin flip
        old_state = coin_states[flip_row * 8 + flip_col]
        new_state = "T" if old_state == "H" else "H"
        new_label = Text(new_state, font_size=12, color=WHITE if new_state == "H" else "#888888")
        new_label.move_to(flip_label.get_center())

        self.play(
            flip_circle.animate.set_stroke(color="#00FF00", width=5),
            Transform(flip_label, new_label),
            run_time=1
        )
        
        # Glowing effect for the flipped coin
        self.play(Indicate(flip_circle, color="#00FF00", scale_factor=1.2))
        self.wait(2)
