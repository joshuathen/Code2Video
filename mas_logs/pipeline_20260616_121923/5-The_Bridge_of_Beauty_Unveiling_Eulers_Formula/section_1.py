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

class Section1Scene(TeachingScene):
    def construct(self):
        # Initialize Layout
        title = "The Grand Encounter"
        lines = [
            'This is the most beautiful formula in all mathematics.',
            'It unites five fundamental constants in one simple bridge.',
            'Meet zero, one, e, i, and pi.',
            'At first, they seem like complete strangers.',
            'Yet they form a single, perfect mathematical family.'
        ]
        self.setup_layout(title, lines)
        
        # Define Symbols using Text to avoid LaTeX dependency issues
        e_sym = Text("e", color=WHITE)
        i_sym = Text("i", color=WHITE)
        pi_sym = Text("π", color=WHITE)
        one_sym = Text("1", color=WHITE)
        zero_sym = Text("0", color=WHITE)
        
        plus_sym = Text("+", color=WHITE)
        equal_sym = Text("=", color=WHITE)
        
        # Bridge asset
        bridge_asset_path = "/mmfs1/data/home/jthen/Code2Video/assets/icon/bridge.svg"
        bridge_icon = SVGMobject(bridge_asset_path).set_color("#ADD8E6")

        # === Animation for Lecture Line 1 ===
        # Five symbols appear at grid positions adjusted to avoid obstruction
        self.lecture[0].set_color(YELLOW)
        
        self.place_at_grid(e_sym, "C1", scale_factor=1.2) # Resolved Issue 32
        self.place_at_grid(i_sym, "D5", scale_factor=1.2) # Resolved Issue 32
        self.place_at_grid(pi_sym, "D2", scale_factor=1.2) # Resolved Issue 33
        self.place_at_grid(one_sym, "E6", scale_factor=1.2)
        self.place_at_grid(zero_sym, "F3", scale_factor=1.2)
        
        self.play(
            FadeIn(e_sym), FadeIn(i_sym), FadeIn(pi_sym), 
            FadeIn(one_sym), FadeIn(zero_sym),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # The symbols glide smoothly into a horizontal line on Row E
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        self.play(
            e_sym.animate.move_to(self.grid["E1"]),
            i_sym.animate.move_to(self.grid["E2"]),
            pi_sym.animate.move_to(self.grid["E3"]),
            one_sym.animate.move_to(self.grid["E4"]),
            zero_sym.animate.move_to(self.grid["E5"]),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Fade in operators and form the formula e^{i pi} + 1 = 0
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Positioning for formula base on Row E and exponents on Row D
        self.place_at_grid(plus_sym, "E3", scale_factor=1.0) # Resolved Issue 34
        self.place_at_grid(equal_sym, "E5", scale_factor=1.0) # Resolved Issue 34
        
        self.play(
            e_sym.animate.move_to(self.grid["E1"]).scale(0.83), # Reset scale to ~1.0
            i_sym.animate.move_to(self.grid["D1"]).scale(0.58), # Scale to ~0.7
            pi_sym.animate.move_to(self.grid["D2"]).scale(0.58), # Scale to ~0.7
            FadeIn(plus_sym),
            one_sym.animate.move_to(self.grid["E4"]).scale(0.83),
            FadeIn(equal_sym),
            zero_sym.animate.move_to(self.grid["E6"]).scale(0.83),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Strangers - draw bridge asset in area A1 to B6
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        
        self.place_in_area(bridge_icon, "A1", "B6", scale_factor=1.5)
        
        self.play(FadeIn(bridge_icon, shift=UP), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Family - scale and change color to gold
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        
        formula_group = VGroup(e_sym, i_sym, pi_sym, plus_sym, one_sym, equal_sym, zero_sym)
        
        self.play(
            formula_group.animate.scale(1.2).set_color("#FFD700"),
            bridge_icon.animate.scale(1.1).set_color("#FFD700"),
            run_time=2
        )
        self.wait(2)
