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

class Section2Scene(TeachingScene):
    def construct(self):
        title = "Prerequisite Knowledge: The Parity Bit"
        lines = [
            "- A parity bit ensures the total ones count is even.",
            "- If data changes, the parity bit becomes odd, signaling error.",
            "- However, parity bits detect errors but cannot locate them."
        ]
        self.setup_layout(title, lines)

        # Initialize lecture colors to GRAY
        for line in self.lecture:
            line.set_color(GRAY)

        # === Animation for Lecture Line 1 ===
        # Show data bits '1, 0, 1' and a box for the parity bit labeled 'P'. (Color: #FFFFFF)
        self.play(self.lecture[0].animate.set_color(WHITE))
        
        bit1 = Text("1", color=WHITE)
        bit2 = Text("0", color=WHITE)
        bit3 = Text("1", color=WHITE)
        
        # Shifted right to avoid lecture line (Issue 30)
        self.place_at_grid(bit1, "B3")
        self.place_at_grid(bit2, "B4")
        self.place_at_grid(bit3, "B5")
        
        # Parity box in Col 6, scaled to avoid clipping (L003, Issue 30)
        parity_box = Square(side_length=0.8, color=WHITE)
        self.place_at_grid(parity_box, "B6", scale_factor=0.8)
        
        # Label 'P' in Col 6 (A6), scaled (L003, Issue 30)
        p_label = Text("P", font_size=24, color=WHITE)
        self.place_at_grid(p_label, "A6", scale_factor=0.8)
        
        self.play(
            FadeIn(bit1),
            FadeIn(bit2),
            FadeIn(bit3),
            Create(parity_box),
            Write(p_label)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Calculate parity: 1+0+1=2 (even), so '0' appears in the 'P' box in green. (Color: #00FF00)
        self.play(
            self.lecture[0].animate.set_color(GRAY), 
            self.lecture[1].animate.set_color("#00FF00")
        )
        
        # Calculation text moved to D3-D6 to avoid Row C overcrowding (Issue 31, 32)
        calc_text = MathTex("1", "+", "0", "+", "1", "=", "2", color="#00FF00")
        self.place_in_area(calc_text, "D3", "D6", scale_factor=0.8)
        
        parity_val = Text("0", color="#00FF00")
        self.place_at_grid(parity_val, "B6", scale_factor=0.8)
        
        self.play(Write(calc_text))
        self.play(FadeIn(parity_val))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Flip one data bit to '1' (becoming '1, 1, 1') and turn the total sum '3' red to signal an error. (Color: #FF0000)
        self.play(
            self.lecture[1].animate.set_color(GRAY), 
            self.lecture[2].animate.set_color("#FF0000")
        )
        
        bit2_red = Text("1", color="#FF0000")
        self.place_at_grid(bit2_red, "B4")
        
        # Update calculation in same area D3-D6 (Issue 32)
        calc_text_new = MathTex("1", "+", "1", "+", "1", "=", "3", color="#FF0000")
        self.place_in_area(calc_text_new, "D3", "D6", scale_factor=0.8)
        
        # Indicate the error (L004)
        self.play(
            Transform(bit2, bit2_red),
            Transform(calc_text, calc_text_new)
        )
        self.play(Indicate(calc_text))
        self.wait(2)
