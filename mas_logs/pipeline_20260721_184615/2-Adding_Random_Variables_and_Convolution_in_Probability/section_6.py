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

class Section6Scene(TeachingScene):
    def construct(self):
        title_text = "Summary & Key Takeaway"
        lecture_lines = [
            "Adding variables means summing probabilities along diagonal lines.",
            "Convolution is the engine driving this mathematical transformation.",
            "It explains why the bell curve appears everywhere."
        ]
        self.setup_layout(title_text, lecture_lines)

        # === Animation for Lecture Line 1 ===
        # Show the yellow diagonal line (#FFFF00) on a 2D plane.
        self.lecture[0].set_color("#FFFF00")
        
        axes = Axes(
            x_range=[0, 4, 1],
            y_range=[0, 4, 1],
            x_length=3,
            y_length=3,
            axis_config={
                "color": "#FFFFFF",
                "stroke_width": 2,
                "include_tip": True,
            }
        )
        
        diag_line = Line(
            start=axes.c2p(3, 0),
            end=axes.c2p(0, 3),
            color="#FFFF00",
            stroke_width=6
        )
        
        diag_visual = VGroup(axes, diag_line)
        # Fix for Issue 39: Move to top-right area (A2-C5) for better spatial distribution
        self.place_in_area(diag_visual, 'A2', 'C5', scale_factor=0.7)
        
        self.play(Create(axes), run_time=1)
        self.play(Create(diag_line), run_time=1)
        self.wait(1.5)

        # === Animation for Lecture Line 2 ===
        # Display 'f * g' in cyan (#00FFFF) with a glowing effect.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color("#00FFFF")
        
        conv_formula = MathTex(r"(f * g)(z)", color="#00FFFF")
        # Fix for Issue 40: Set scale to 1.0 and move to middle area (D2-D5)
        self.place_in_area(conv_formula, 'D2', 'D5', scale_factor=1.0)
        
        self.play(Write(conv_formula))
        self.play(Indicate(conv_formula, color="#00FFFF", scale_factor=1.1))
        self.wait(1.5)

        # === Animation for Lecture Line 3 ===
        # Show a sequence: Square -> Triangle -> Bell Curve (#FFFFFF).
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color("#FFFFFF")
        
        # Fix for Issue 41: Move sequence to bottom area (E2-F5) to align with lecture lines
        # Square (Uniform)
        square = Square(side_length=1.5, color="#FFFFFF", fill_opacity=0.3)
        self.place_in_area(square, 'E2', 'F5', scale_factor=0.6)
        
        # Triangle (Sum of two uniforms)
        tri_points = [
            square.get_corner(DL),
            square.get_corner(DR),
            square.get_top()
        ]
        triangle = Polygon(*tri_points, color="#FFFFFF", fill_opacity=0.3)
        
        # Bell Curve (Normal distribution)
        bell_curve = FunctionGraph(
            lambda x: 1.2 * np.exp(-1.5 * x**2),
            x_range=[-2, 2],
            color="#FFFFFF"
        )
        self.place_in_area(bell_curve, 'E2', 'F5', scale_factor=0.8)

        self.play(Create(square))
        self.wait(1.0)
        self.play(ReplacementTransform(square, triangle))
        self.wait(1.0)
        self.play(ReplacementTransform(triangle, bell_curve))
        self.wait(2.0)
