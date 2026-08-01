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

class Section4Scene(TeachingScene):
    def construct(self):
        # Initializing the layout for Section 4
        self.setup_layout(
            "The Solution: L'H\u00f4pital's Rule",
            [
                "L'Hôpital's Rule compares how fast the functions change.",
                "Imagine two cars racing toward the same finish line.",
                "Both reach zero, but their speeds determine the ratio.",
                "We compare the derivatives of the top and bottom.",
                "The limit of their speeds reveals the final answer."
            ]
        )

        # === Animation for Lecture Line 1 ===
        # Display L'Hopital's Rule formula: lim f(x)/g(x) = lim f'(x)/g'(x) in #FFFFFF
        formula = Text("lim f(x)/g(x) = lim f'(x)/g'(x)", color="#FFFFFF", font_size=24)
        self.place_at_grid(formula, "B3", scale_factor=1.0)
        
        self.play(
            self.lecture[0].animate.set_color("#FFFFFF"),
            Write(formula)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Show two vertical bars, Green (f) and Blue (g)
        bar_f = Rectangle(height=2.0, width=0.4, color="#00FF00", fill_opacity=0.8)
        bar_g = Rectangle(height=2.5, width=0.4, color="#00BFFF", fill_opacity=0.8)
        self.place_at_grid(bar_f, "D2")
        self.place_at_grid(bar_g, "D4")
        
        label_f = Text("f", color="#00FF00", font_size=24)
        label_g = Text("g", color="#00BFFF", font_size=24)
        self.place_at_grid(label_f, "C2")
        self.place_at_grid(label_g, "C4")
        
        self.play(
            self.lecture[1].animate.set_color("#00FF00"),
            Create(bar_f), Create(bar_g),
            Write(label_f), Write(label_g)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Both reach zero, heights decrease
        self.play(
            self.lecture[2].animate.set_color("#00BFFF"),
            bar_f.animate.stretch_to_fit_height(0.1, about_edge=DOWN),
            bar_g.animate.stretch_to_fit_height(0.1, about_edge=DOWN),
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Compare derivatives (speeds). Use specific sin(x)/x logic for math steps.
        
        # Hide previous bars/labels to make room for math steps
        self.play(FadeOut(bar_f), FadeOut(bar_g), FadeOut(label_f), FadeOut(label_g))

        lhopital_step = Text("lim sin(x)/x = lim cos(x)/1", color="#FFFF00", font_size=24)
        self.place_in_area(lhopital_step, 'C2', 'C6', scale_factor=0.9) # [Issue 31 Fix]

        derivative_step = Text("= cos(0) / 1", color="#00FF00", font_size=24)
        self.place_in_area(derivative_step, 'D2', 'D5', scale_factor=1.0) # [Issue 32 Fix]
        
        # Tangent arrows representing derivatives (speeds)
        arrow_f = Arrow(start=ORIGIN, end=UP*1.0, color="#00FF00").scale(0.8)
        arrow_g = Arrow(start=ORIGIN, end=UP*1.0, color="#00BFFF").scale(0.8)
        self.place_at_grid(arrow_f, "C1")
        self.place_at_grid(arrow_g, "D1")
        
        label_f_p = Text("f'", color="#00FF00", font_size=20)
        label_g_p = Text("g'", color="#00BFFF", font_size=20)
        self.place_at_grid(label_f_p, "B1")
        self.place_at_grid(label_g_p, "E1")

        self.play(
            self.lecture[3].animate.set_color("#FFFF00"),
            Create(arrow_f), Create(arrow_g),
            Write(label_f_p), Write(label_g_p)
        )
        self.play(Write(lhopital_step))
        self.play(FadeIn(derivative_step, shift=DOWN))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Show final limit result
        final_answer = Text("= 1", color="#FFFFFF", font_size=32)
        self.place_at_grid(final_answer, 'E3', scale_factor=1.1) # [Issue 33 Fix]
        
        self.play(
            self.lecture[4].animate.set_color("#FFFFFF"),
            Write(final_answer)
        )
        self.wait(2)
