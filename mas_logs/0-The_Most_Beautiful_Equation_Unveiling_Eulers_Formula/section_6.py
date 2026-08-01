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

class Section6Scene(TeachingScene):
    def construct(self):
        # Setup layout with title and lecture lines
        self.setup_layout("Conclusion: The Unity of Math", [
            "This elegant identity links algebra, geometry, and calculus.",
            "It reveals a deep, hidden symmetry within the universe.",
            "One line of ink connects the most fundamental concepts."
        ])

        # FIX: Replaced MathTex with MarkupText to avoid FileNotFoundError: 'latex'
        # MarkupText renders math-like formatting using Pango/Unicode without requiring a LaTeX distribution.
        formula = MarkupText("e<sup>πi</sup> + 1 = 0", color="#FFD700")
        alg_label = Text("Algebra", font_size=24, color="#FFFFFF")
        geo_label = Text("Geometry", font_size=24, color="#FFFFFF")
        calc_label = Text("Calculus", font_size=24, color="#FFFFFF")

        # === Animation for Lecture Line 1 ===
        # Highlight first lecture line to match formula color
        self.play(self.lecture[0].animate.set_color("#FFD700"))

        # Position elements using the grid system
        # Identity in the central area
        self.place_in_area(formula, "C3", "D4", scale_factor=1.5)
        # Labels surrounding the identity within 1 grid unit to avoid occlusion and show unity
        self.place_at_grid(alg_label, "B3") # Above
        self.place_at_grid(geo_label, "C2") # Left
        self.place_at_grid(calc_label, "C5") # Right

        self.play(Write(formula))
        self.play(
            FadeIn(alg_label, shift=UP * 0.2),
            FadeIn(geo_label, shift=RIGHT * 0.2),
            FadeIn(calc_label, shift=LEFT * 0.2)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight second lecture line to match connection lines
        self.play(self.lecture[1].animate.set_color("#FFFFFF"))

        # Create connecting lines representing the hidden symmetry/unity
        line1 = Line(alg_label.get_bottom(), formula.get_top(), color="#FFFFFF", stroke_width=3)
        line2 = Line(geo_label.get_right(), formula.get_left(), color="#FFFFFF", stroke_width=3)
        line3 = Line(calc_label.get_left(), formula.get_right(), color="#FFFFFF", stroke_width=3)

        self.play(Create(line1), Create(line2), Create(line3))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Highlight third lecture line
        self.play(self.lecture[2].animate.set_color("#FFD700"))

        # Clean up the scene to leave the fundamental formula on a black background
        self.play(
            FadeOut(alg_label), FadeOut(geo_label), FadeOut(calc_label),
            FadeOut(line1), FadeOut(line2), FadeOut(line3)
        )
        
        # Final focus on the elegance of Euler's identity
        self.play(Indicate(formula, color="#FFD700", scale_factor=1.2))
        self.wait(3)
