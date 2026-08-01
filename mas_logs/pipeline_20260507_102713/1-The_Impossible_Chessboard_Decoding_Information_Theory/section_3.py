from manim import *
import numpy as np

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

class Section3Scene(TeachingScene):
    def construct(self):
        lecture_lines_text = [
            "Every 'Heads' coin contributes its position to the board.",
            "Bob finds the signature by XORing these positions.",
            "Start with zero and XOR every position showing 'Heads'.",
            "This six-bit number is the current board's identity.",
            "One coin flip completely changes this unique signature."
        ]
        self.setup_layout("Calculating the 'Board State'", lecture_lines_text)

        # Assets
        coin_path = "/mmfs1/data/home/jthen/Code2Video/assets/icon/coin.svg"
        
        # === Setup Objects (Not yet added) ===
        # Board Setup (8x8 Chessboard)
        board_squares = VGroup()
        for i in range(64):
            sq_color = "#333333" if (i // 8 + i % 8) % 2 == 0 else "#222222"
            sq = Square(side_length=0.4, fill_opacity=1, fill_color=sq_color, stroke_width=0.5, stroke_color=GRAY)
            board_squares.add(sq)
        
        board_group = VGroup(*board_squares).arrange_in_grid(8, 8, buff=0)
        self.place_in_area(board_group, "A1", "D4", scale_factor=1.0)
        
        # Define Heads positions
        heads_indices = [13, 25, 42]
        heads_coins = VGroup()
        heads_highlights = VGroup()
        binary_labels = VGroup()

        for idx in heads_indices:
            # [Asset: /mmfs1/data/home/jthen/Code2Video/assets/icon/coin.svg]
            coin = SVGMobject(coin_path)
            coin.set(height=0.3)
            coin.move_to(board_group[idx].get_center())
            heads_coins.add(coin)
            
            highlight = Square(side_length=0.4, stroke_color="#FFD700", stroke_width=2)
            highlight.move_to(board_group[idx].get_center())
            heads_highlights.add(highlight)
            
            bin_str = f"{idx:06b}"
            bin_label = Text(bin_str, font_size=14, color="#FFD700")
            # Label near the square
            bin_label.next_to(board_group[idx], UP, buff=0.05)
            binary_labels.add(bin_label)

        # === Animation for Lecture Line 1 ===
        # "Every 'Heads' coin contributes its position to the board."
        self.play(self.lecture[0].animate.set_color("#FFD700"))
        self.play(
            Create(board_group),
            FadeIn(heads_coins),
            Create(heads_highlights),
            Write(binary_labels),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "Bob finds the signature by XORing these positions."
        # Move binary indices of 'Heads' squares into a vertical column for XORing.
        math_col_labels = binary_labels.copy()
        
        # Targets for the column on the right (Grid column 5)
        targets = [self.grid["A5"], self.grid["B5"], self.grid["C5"]]

        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color("#FFD700"),
            *[math_col_labels[i].animate.move_to(targets[i]) for i in range(len(targets))],
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "Start with zero and XOR every position showing 'Heads'."
        # Animate a running XOR calculation starting from 000000.
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color("#FFD700")
        )
        
        running_xor_val = 0
        xor_display = Text(f"Base: {running_xor_val:06b}", font_size=20, color=WHITE)
        self.place_at_grid(xor_display, "E5")
        self.play(Write(xor_display))
        
        xor_symbol = Text("⊕", font_size=20, color=WHITE)
        
        for i, idx in enumerate(heads_indices):
            running_xor_val = running_xor_val ^ idx
            
            # Show symbol next to label
            op_sym = xor_symbol.copy().next_to(math_col_labels[i], LEFT, buff=0.1)
            self.play(Write(op_sym), run_time=0.4)
            
            new_xor_display = Text(f"Result: {running_xor_val:06b}", font_size=20, color=WHITE).move_to(xor_display)
            self.play(Transform(xor_display, new_xor_display), run_time=0.6)
            self.wait(0.3)

        # === Animation for Lecture Line 4 ===
        # "This six-bit number is the current board's identity."
        # Display the final XOR result in a box labeled 'Board Signature' (#1E90FF).
        self.play(
            self.lecture[2].animate.set_color(WHITE),
            self.lecture[3].animate.set_color("#1E90FF")
        )

        sig_box = SurroundingRectangle(xor_display, color="#1E90FF", buff=0.2)
        sig_label = Text("Board Signature", font_size=16, color="#1E90FF")
        self.place_at_grid(sig_label, "F5")
        
        self.play(Create(sig_box), Write(sig_label))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # "One coin flip completely changes this unique signature."
        # Alice flips one coin [Asset: /mmfs1/data/home/jthen/Code2Video/assets/icon/coin.svg].
        # Show 'Board Signature' value instantly changing to a new number.
        self.play(
            self.lecture[3].animate.set_color(WHITE),
            self.lecture[4].animate.set_color("#FFD700")
        )

        flip_idx = 5 # Square 5
        flip_coin = SVGMobject(coin_path)
        flip_coin.set(height=0.3)
        flip_coin.move_to(board_group[flip_idx].get_center())
        
        flip_highlight = Square(side_length=0.4, stroke_color="#FF4500", stroke_width=3)
        flip_highlight.move_to(board_group[flip_idx].get_center())
        
        # New calculation
        new_val = running_xor_val ^ flip_idx
        updated_sig_text = Text(f"Result: {new_val:06b}", font_size=20, color="#FFD700").move_to(xor_display)
        
        self.play(
            FadeIn(flip_coin), 
            Create(flip_highlight)
        )
        
        flash = Flash(xor_display, color="#FFD700", run_time=0.5)
        self.play(
            Transform(xor_display, updated_sig_text),
            flash
        )
        
        self.wait(2)
