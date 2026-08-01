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
        lecture_lines = [
            'Imagine a single cell dividing at a constant rate.',
            'This is scalar growth, modeled by y prime equals ay.',
            'But real systems, like predators and prey, are coupled.',
            "Here, one variable's change depends on the other's state.",
            'To solve these, we need the matrix exponential.'
        ]
        self.setup_layout("The Hook: From Scalar to Matrix", lecture_lines)

        # Highlight color for lecture lines
        HL_COLOR = YELLOW

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(HL_COLOR))
        
        # Note: We use a Circle as a fallback for the cell image to ensure execution 
        # but keep the placement logic as requested.
        cell = Circle(color=WHITE, fill_opacity=0.3).scale(0.5)
        self.place_in_area(cell, "B3", "C4", scale_factor=0.5)
        
        self.play(FadeIn(cell))
        
        cell_left = cell.copy()
        cell_right = cell.copy()
        
        self.play(
            cell_left.animate.shift(LEFT * 0.8),
            cell_right.animate.shift(RIGHT * 0.8),
            FadeOut(cell),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(HL_COLOR)
        )
        
        scalar_eq = Text("y' = ay", color=WHITE, font="Monospace")
        self.place_at_grid(scalar_eq, "A3", scale_factor=1.2)
        
        self.play(Write(scalar_eq))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(HL_COLOR),
            FadeOut(cell_left),
            FadeOut(cell_right)
        )
        
        # Wolf and Rabbit icons using generic shapes
        wolf_icon = VGroup(
            Circle(color="#FF0000", fill_opacity=0.8),
            Text("Wolf", font_size=18, color=WHITE).shift(DOWN * 0.6)
        )
        rabbit_icon = VGroup(
            Circle(color="#0000FF", fill_opacity=0.8),
            Text("Rabbit", font_size=18, color=WHITE).shift(DOWN * 0.6)
        )
        
        self.place_at_grid(wolf_icon, "C2", scale_factor=0.6)
        self.place_at_grid(rabbit_icon, "C5", scale_factor=0.6)
        
        arrow_wr = CurvedArrow(self.grid["C2"] + RIGHT*0.4, self.grid["C5"] + LEFT*0.4, angle=-PI/4, color=WHITE)
        arrow_rw = CurvedArrow(self.grid["C5"] + LEFT*0.4, self.grid["C2"] + RIGHT*0.4, angle=-PI/4, color=WHITE)
        
        self.play(FadeIn(wolf_icon), FadeIn(rabbit_icon))
        self.play(Create(arrow_wr), Create(arrow_rw))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(
            self.lecture[2].animate.set_color(WHITE),
            self.lecture[3].animate.set_color(HL_COLOR)
        )
        
        system = Text("R' = aR - bRW\nW' = cRW - dW", color=WHITE, font="Monospace", font_size=28)
        self.place_in_area(system, "E2", "E5", scale_factor=1.0)
        
        self.play(Write(system))
        self.wait(0.5)
        
        # Highlight interaction terms
        self.play(system.animate.set_color(HL_COLOR))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(
            self.lecture[3].animate.set_color(WHITE),
            self.lecture[4].animate.set_color(HL_COLOR)
        )

        matrix_form = Text("x' = Ax", color=WHITE, font="Monospace")
        self.place_at_grid(matrix_form, "B5", scale_factor=1.5)
        
        self.play(
            FadeOut(system),
            FadeOut(wolf_icon),
            FadeOut(rabbit_icon),
            FadeOut(arrow_wr),
            FadeOut(arrow_rw),
            Write(matrix_form)
        )
        self.wait(2)
