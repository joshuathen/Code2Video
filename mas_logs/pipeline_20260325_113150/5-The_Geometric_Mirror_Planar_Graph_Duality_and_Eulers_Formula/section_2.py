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
        # Initialize the layout with the title and lecture lines
        # Using mandatory strings from instructions
        self.setup_layout("The Constant Harmony: Euler's Characteristic Formula", [
            "Euler’s formula relates vertices, edges, and faces for planar graphs.",
            "A triangle satisfies the formula: three minus three plus two.",
            "Adding a bridge vertex and edge keeps the equation balanced.",
            "Adding an edge creates a new face, maintaining the result.",
            "The characteristic value of two remains constant for all configurations."
        ])

        # Colors as per instructions
        v_color = "#00FF00"  # Vertices
        e_color = "#FFFFFF"  # Edges
        f_color = "#FFFF00"  # Faces
        res_color = "#FFD700" # Golden Glow

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(v_color))
        
        # Display the main formula V - E + F = 2
        formula = VGroup(
            Text("V", font_size=42), Text("-", font_size=42), Text("E", font_size=42),
            Text("+", font_size=42), Text("F", font_size=42), Text("=", font_size=42),
            Text("2", font_size=42)
        ).arrange(RIGHT, buff=0.15)
        
        formula[0].set_color(v_color)
        formula[2].set_color(e_color)
        formula[4].set_color(f_color)
        # Fix for Issue #24: added scale_factor=0.8 to formula placement
        self.place_in_area(formula, "A2", "A5", scale_factor=0.8)
        self.play(Write(formula))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(e_color)
        )

        # Create the Triangle Graph
        v1 = Dot(self.grid["C3"], color=v_color)
        v2 = Dot(self.grid["E2"], color=v_color)
        v3 = Dot(self.grid["E4"], color=v_color)
        
        e1 = Line(self.grid["C3"], self.grid["E2"], color=e_color)
        e2 = Line(self.grid["E2"], self.grid["E4"], color=e_color)
        e3 = Line(self.grid["E4"], self.grid["C3"], color=e_color)
        
        face1 = Polygon(self.grid["C3"], self.grid["E2"], self.grid["E4"], 
                        fill_opacity=0.3, fill_color=f_color, stroke_width=0)

        self.play(FadeIn(v1, v2, v3))
        self.play(Create(e1), Create(e2), Create(e3))
        self.play(FadeIn(face1))

        # Calculation line: 3 - 3 + 2 = 2
        calc1 = VGroup(
            Text("3", font_size=36), Text("-", font_size=36), Text("3", font_size=36),
            Text("+", font_size=36), Text("2", font_size=36), Text("=", font_size=36),
            Text("2", font_size=36)
        ).arrange(RIGHT, buff=0.15)
        
        calc1[0].set_color(v_color)
        calc1[2].set_color(e_color)
        calc1[4].set_color(f_color)
        # Fix for Issue #25: added scale_factor=0.8 to calc1 placement
        self.place_in_area(calc1, "B2", "B5", scale_factor=0.8)
        self.play(Write(calc1))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(f_color)
        )

        v4 = Dot(self.grid["E6"], color=v_color)
        e4 = Line(self.grid["E4"], self.grid["E6"], color=e_color)

        self.play(FadeIn(v4))
        self.play(Create(e4))

        # Update calculation line to: 4 - 4 + 2 = 2
        calc2 = VGroup(
            Text("4", font_size=36), Text("-", font_size=36), Text("4", font_size=36),
            Text("+", font_size=36), Text("2", font_size=36), Text("=", font_size=36),
            Text("2", font_size=36)
        ).arrange(RIGHT, buff=0.15)
        
        calc2[0].set_color(v_color)
        calc2[2].set_color(e_color)
        calc2[4].set_color(f_color)
        # Applying scale factor here as well for consistency with Issue #25 and #26
        self.place_in_area(calc2, "B2", "B5", scale_factor=0.8)
        
        self.play(ReplacementTransform(calc1, calc2))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(
            self.lecture[2].animate.set_color(WHITE),
            self.lecture[3].animate.set_color(v_color)
        )

        e5 = Line(self.grid["E6"], self.grid["C3"], color=e_color)
        face2 = Polygon(self.grid["E4"], self.grid["E6"], self.grid["C3"], 
                        fill_opacity=0.3, fill_color=f_color, stroke_width=0)

        self.play(Create(e5))
        self.play(FadeIn(face2))

        # Update calculation line to: 4 - 5 + 3 = 2
        calc3 = VGroup(
            Text("4", font_size=36), Text("-", font_size=36), Text("5", font_size=36),
            Text("+", font_size=36), Text("3", font_size=36), Text("=", font_size=36),
            Text("2", font_size=36)
        ).arrange(RIGHT, buff=0.15)
        
        calc3[0].set_color(v_color)
        calc3[2].set_color(e_color)
        calc3[4].set_color(f_color)
        # Fix for Issue #26: added scale_factor=0.8 to calc3 placement
        self.place_in_area(calc3, "B2", "B5", scale_factor=0.8)

        self.play(ReplacementTransform(calc2, calc3))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(
            self.lecture[3].animate.set_color(WHITE),
            self.lecture[4].animate.set_color(res_color)
        )
        
        # Pulse the final '2' and the '2' in the main formula
        final_two_calc = calc3[6]
        final_two_formula = formula[6]
        self.play(
            final_two_calc.animate.set_color(res_color).scale(1.5),
            final_two_formula.animate.set_color(res_color).scale(1.5),
            run_time=0.5
        )
        self.play(
            final_two_calc.animate.scale(1/1.5),
            final_two_formula.animate.scale(1/1.5),
            run_time=0.5
        )
        self.wait(3)
