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
        lecture_lines = [
            "Observe the pattern as we reduce the factorial.",
            "To move down, we divide by the current number.",
            "Three factorial divided by three gives two factorial.",
            "Two factorial divided by two gives one factorial.",
            "One factorial divided by one shows zero factorial is one."
        ]
        self.setup_layout("The Pattern Approach (Algebraic Logic)", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFFF00")
        
        # Replaced MathTex with Text due to environment missing latex
        eq4 = Text("4! = 24", color="#FFFFFF")
        eq3 = Text("3! = 6", color="#FFFFFF")
        eq2 = Text("2! = 2", color="#FFFFFF")
        eq1 = Text("1! = 1", color="#FFFFFF")
        
        self.place_at_grid(eq4, 'A3', scale_factor=0.9)
        self.place_at_grid(eq3, 'B3', scale_factor=0.9)
        self.place_at_grid(eq2, 'C3', scale_factor=0.9)
        self.place_at_grid(eq1, 'D3', scale_factor=0.9)
        
        self.play(Write(VGroup(eq4, eq3, eq2, eq1)))
        self.wait(1.5)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color("#FFFFFF")
        self.lecture[1].set_color("#FFFF00")
        
        # Curved yellow arrow from 24 (A3) down to 6 (B3)
        arrow1 = CurvedArrow(self.grid['A4'], self.grid['B4'], angle=-TAU/4, color="#FFFF00")
        label1 = Text("÷ 4", color="#FFFF00")
        self.place_in_area(label1, 'A5', 'B5', scale_factor=0.8)
        
        self.play(Create(arrow1), Write(label1))
        self.wait(1.5)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color("#FFFFFF")
        self.lecture[2].set_color("#FFFF00")
        
        # Curved yellow arrow from 6 (B3) down to 2 (C3)
        arrow2 = CurvedArrow(self.grid['B4'], self.grid['C4'], angle=-TAU/4, color="#FFFF00")
        label2 = Text("÷ 3", color="#FFFF00")
        self.place_in_area(label2, 'B5', 'C5', scale_factor=0.8)
        
        self.play(Create(arrow2), Write(label2))
        self.wait(1.5)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color("#FFFFFF")
        self.lecture[3].set_color("#FFFF00")
        
        # Curved yellow arrow from 2 (C3) down to 1 (D3)
        arrow3 = CurvedArrow(self.grid['C4'], self.grid['D4'], angle=-TAU/4, color="#FFFF00")
        label3 = Text("÷ 2", color="#FFFF00")
        self.place_in_area(label3, 'C5', 'D5', scale_factor=0.8)
        
        self.play(Create(arrow3), Write(label3))
        self.wait(1.5)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color("#FFFFFF")
        self.lecture[4].set_color("#FFFF00")
        
        # Final line 0! = 1. Used VGroup to keep indexing logic for Flash.
        eq0 = VGroup(Text("0! =", color="#FFFFFF"), Text("1", color="#FFFFFF")).arrange(RIGHT, buff=0.15)
        self.place_at_grid(eq0, 'E3', scale_factor=0.9)
        
        # Final arrow from 1! to 0!
        arrow4 = CurvedArrow(self.grid['D4'], self.grid['E4'], angle=-TAU/4, color="#FFFF00")
        label4 = Text("÷ 1", color="#FFFF00")
        self.place_in_area(label4, 'D5', 'E5', scale_factor=0.8)
        
        self.play(
            Write(eq0),
            Create(arrow4),
            Write(label4)
        )
        self.wait(0.5)
        
        # Flash result 1 in green
        self.play(Flash(eq0[1], color="#00FF00", flash_radius=0.4))
        eq0[1].set_color("#00FF00")
        
        self.wait(3)
        self.lecture[4].set_color("#FFFFFF")
        self.wait(1)
