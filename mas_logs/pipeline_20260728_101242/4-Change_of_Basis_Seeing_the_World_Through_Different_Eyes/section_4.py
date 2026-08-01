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
        # Setup layout with lecture lines from storyboard
        self.setup_layout("The Mathematical Transformation", [
            "- Multiply the matrix to convert coordinates between bases.",
            "- Tilted grid lines align with the standard square grid.",
            "- The transformation reveals how the other person sees it."
        ])
        
        # Colors
        formula_white = "#FFFFFF"
        matrix_cyan = "#00FFFF"
        owl_yellow = "#FFFF00"
        point_green = "#00FF00"
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(matrix_cyan))
        
        # Display the formula '[x]_Human = P * [x]_Owl' in #FFFFFF and highlight P in #00FFFF.
        formula = MathTex(
            r"[\vec{x}]_{\text{Human}}", "=", "P", r"\cdot [\vec{x}]_{\text{Owl}}",
            font_size=36, color=formula_white
        )
        formula[2].set_color(matrix_cyan) # Highlight P
        
        self.place_in_area(formula, 'A2', 'A5', scale_factor=1.0)
        self.play(Write(formula))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[0].animate.set_color(WHITE), self.lecture[1].animate.set_color(owl_yellow))
        
        # Create Owl's tilted grid #FFFF00
        # Matrix P from Section 3 was [[1, -1], [1, 1]]
        # The grid area is roughly B2 to F6.
        tilted_grid = NumberPlane(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            x_length=4,
            y_length=4,
            background_line_style={"stroke_opacity": 0.8, "stroke_color": owl_yellow},
            axis_config={"stroke_color": owl_yellow}
        )
        self.place_in_area(tilted_grid, 'B2', 'F6', scale_factor=1.0)
        tilted_grid.apply_matrix([[1, -1], [1, 1]])
        
        # Show the Owl's tilted grid
        self.play(Create(tilted_grid))
        self.wait(0.5)
        
        # Prepare standard square grid (Human)
        standard_grid = NumberPlane(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            x_length=4,
            y_length=4,
            background_line_style={"stroke_opacity": 0.4, "stroke_color": WHITE},
            axis_config={"stroke_color": WHITE}
        )
        self.place_in_area(standard_grid, 'B2', 'F6', scale_factor=1.0)
        
        # Animate it unwarping into the standard square grid.
        self.play(Transform(tilted_grid, standard_grid), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[1].animate.set_color(WHITE), self.lecture[2].animate.set_color(point_green))
        
        # Plot a point moving from (1,1) Owl to (0,2) Human; highlight final coordinates in #00FF00.
        # We start by bringing back the tilted context for the point.
        self.play(FadeOut(tilted_grid))
        
        # Re-create tilted grid for context
        context_owl_grid = NumberPlane(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            x_length=4,
            y_length=4,
            background_line_style={"stroke_opacity": 0.8, "stroke_color": owl_yellow}
        )
        self.place_in_area(context_owl_grid, 'B2', 'F6', scale_factor=1.0)
        context_owl_grid.apply_matrix([[1, -1], [1, 1]])
        
        # Point at Owl (1, 1) which is physically (0, 2) in the scene's standard coordinates.
        target_point_phys = context_owl_grid.c2p(1, 1)
        dot = Dot(target_point_phys, color=point_green)
        
        # Label for Owl coordinates
        owl_label = MathTex(r"(1, 1)_{\text{Owl}}", font_size=24, color=owl_yellow)
        owl_label.next_to(dot, UR, buff=0.1)
        
        self.play(Create(context_owl_grid), FadeIn(dot), Write(owl_label))
        self.wait(1)
        
        # Prepare Human grid and label
        context_human_grid = NumberPlane(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            x_length=4,
            y_length=4,
            background_line_style={"stroke_opacity": 0.4, "stroke_color": WHITE}
        )
        self.place_in_area(context_human_grid, 'B2', 'F6', scale_factor=1.0)
        
        human_label = MathTex(r"(0, 2)_{\text{Human}}", font_size=24, color=point_green)
        human_label.next_to(dot, UR, buff=0.1)
        
        # Transition to Human view
        self.play(
            Transform(context_owl_grid, context_human_grid),
            ReplacementTransform(owl_label, human_label),
            run_time=2
        )
        self.play(Indicate(human_label, color=point_green))
        
        self.wait(2)
        # Clear color
        self.play(self.lecture[2].animate.set_color(WHITE))
