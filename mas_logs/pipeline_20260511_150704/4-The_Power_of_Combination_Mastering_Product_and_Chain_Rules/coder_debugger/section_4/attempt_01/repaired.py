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
        self.lecture = VGroup(*lecture_texts).arrange(DOWN, aligned_edge=LEFT, buff=0.4).scale(0.8)
        self.lecture.to_edge(LEFT, buff=0.5)
        self.add(self.lecture)

        # Define fine-grained animation grid (6x6 grid on right side)
        self.grid = {}
        rows = ["A", "B", "C", "D", "E", "F"]  # Top to bottom
        cols = ["1", "2", "3", "4", "5", "6"]  # Left to right

        for i, row in enumerate(rows):
            for j, col in enumerate(cols):
                # Offset grid to the right side of the screen
                x = 1.0 + j * 1.0
                y = 2.0 - i * 0.8
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

class Section4Scene(TeachingScene):
    def construct(self):
        # Setup the layout with specific lines
        self.setup_layout("The Synergy: Combining Both Rules", [
            'Complex functions often require using both rules together.',
            'Start with the Product Rule to split the components.',
            'Apply the Chain Rule when differentiating the nested term.',
            'Track each layer carefully to avoid calculation errors.',
            'Recombine the branches to find the final derivative.'
        ])

        # Hide lecture initially to animate them one by one
        for line in self.lecture:
            line.set_opacity(0.3)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_opacity(1).set_color(YELLOW))
        func_h = Text("h(x) = x² · sin(3x)", color=WHITE, font_size=24)
        self.place_in_area(func_h, "A1", "A3")
        self.play(Write(func_h))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_opacity(1).set_color(YELLOW))
        prod_step = Text("u = x²,  v = sin(3x)", color=BLUE, font_size=22)
        self.place_in_area(prod_step, "B1", "B3")
        self.play(FadeIn(prod_step, shift=RIGHT))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_opacity(1).set_color(YELLOW))
        chain_step = Text("v' = cos(3x) · 3", color=RED, font_size=22)
        self.place_in_area(chain_step, "C1", "C3")
        self.play(Write(chain_step))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_opacity(1).set_color(YELLOW))
        u_prime = Text("u' = 2x", color=BLUE, font_size=22)
        self.place_in_area(u_prime, "D1", "D3")
        self.play(FadeIn(u_prime, shift=UP))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_opacity(1).set_color(YELLOW))
        final_result = Text("h'(x) = 2x·sin(3x) + x²·3cos(3x)", color=GREEN, font_size=22)
        self.place_in_area(final_result, "E1", "F3")
        self.play(TransformFromCopy(VGroup(func_h, prod_step, chain_step, u_prime), final_result))
        self.wait(3)