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
        # Setup layout with title and lecture lines
        title_text = "The Mystery of the Tangled Equation"
        lecture_lines = [
            "Explicit functions isolate y on one side clearly.",
            "Implicit equations leave x and y tangled together.",
            "Solving for y can create multiple complex pieces.",
            "Implicit differentiation handles these equations without rearranging.",
            "It treats the relationship as a single cohesive rule."
        ]
        self.setup_layout(title_text, lecture_lines)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#00FF00"))
        
        # Using VGroup of Text to bypass LaTeX dependency while maintaining indexing
        eq1 = VGroup(
            Text("y"), 
            Text("="), 
            Text("2x + 3")
        ).arrange(RIGHT, buff=0.15).set_color("#00FF00")
        self.place_in_area(eq1, "A2", "B5", scale_factor=1.5)
        
        # Box around 'y' (eq1[0])
        box_y = SurroundingRectangle(eq1[0], color="#00FF00", buff=0.15)
        
        self.play(Write(eq1))
        self.play(Create(box_y))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FF00FF"))
        
        # eq2: x² + y² = 25 using Unicode superscripts
        eq2 = VGroup(
            Text("x²"), 
            Text("+"), 
            Text("y²"), 
            Text("="), 
            Text("25")
        ).arrange(RIGHT, buff=0.15).set_color("#FF00FF")
        self.place_in_area(eq2, "C2", "D5", scale_factor=1.5)
        
        # Swirling arrow around x² and y²
        arc1 = CurvedArrow(eq2[0].get_top() + 0.1 * UP, eq2[2].get_top() + 0.1 * UP, color="#FF00FF", angle=-TAU/3)
        arc2 = CurvedArrow(eq2[2].get_bottom() + 0.1 * DOWN, eq2[0].get_bottom() + 0.1 * DOWN, color="#FF00FF", angle=-TAU/3)
        swirl = VGroup(arc1, arc2)

        self.play(FadeOut(eq1), FadeOut(box_y))
        self.play(Write(eq2))
        self.play(Create(swirl))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FFFF00"))
        
        # eq3: y = ±√(25 - x²) using Unicode
        eq3 = VGroup(
            Text("y"), 
            Text("="), 
            Text("±"), 
            Text("√(25 - x²)")
        ).arrange(RIGHT, buff=0.15).set_color("#FFFF00")
        self.place_in_area(eq3, "C2", "D5", scale_factor=1.5)
        
        self.play(FadeOut(swirl))
        self.play(ReplacementTransform(eq2, eq3))
        
        # Flash the ± symbol (eq3[2])
        pm_symbol = eq3[2]
        self.play(Flash(pm_symbol, color="#FFFF00", flash_radius=0.4, line_length=0.25))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color("#00FFFF"))
        
        # Return to x² + y² = 25
        eq2_back = VGroup(
            Text("x²"), 
            Text("+"), 
            Text("y²"), 
            Text("="), 
            Text("25")
        ).arrange(RIGHT, buff=0.15).set_color("#00FFFF")
        self.place_in_area(eq2_back, "C2", "D5", scale_factor=1.5)
        
        star = Star(n=5, outer_radius=0.3, inner_radius=0.15, color="#00FFFF", fill_opacity=0.8)
        self.place_at_grid(star, "C5", scale_factor=1.0)
        
        self.play(ReplacementTransform(eq3, eq2_back))
        self.play(FadeIn(star))
        self.play(star.animate.scale(1.5).set_stroke(width=10), rate_func=there_and_back)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color("#FFFFFF"))
        
        outer_border = SurroundingRectangle(eq2_back, color="#FFFFFF", buff=0.4, stroke_width=8)
        
        self.play(FadeOut(star))
        self.play(Create(outer_border))
        self.wait(2)
