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

class Section4Scene(Scene):
    def setup_layout(self, title_text, lecture_lines):
        # Set background color
        self.camera.background_color = "#000000"

        # Create title at the top
        self.title_obj = Text(title_text, font_size=32, color=WHITE).to_edge(UP, buff=0.5)
        self.add(self.title_obj)

        # Create lecture text group on the left side
        self.lecture_group = VGroup(*[
            Text(line, font_size=20, color=WHITE)
            for line in lecture_lines
        ]).arrange(DOWN, aligned_edge=LEFT, buff=0.4)
        self.lecture_group.to_edge(LEFT, buff=0.6)
        self.add(self.lecture_group)

        # Define grid for visual placement on the right side of the screen
        self.grid_positions = {}
        rows = ["A", "B", "C", "D", "E", "F"]
        cols = ["1", "2", "3", "4", "5", "6"]

        # Center the grid area horizontally to the right
        start_x = 1.0
        start_y = 1.5
        for i, row in enumerate(rows):
            for j, col in enumerate(cols):
                # Calculating grid coordinates
                x = start_x + (j * 0.9)
                y = start_y - (i * 0.9)
                self.grid_positions[f"{row}{col}"] = np.array([x, y, 0])

    def place_at_grid(self, mobject, grid_pos, scale_factor=1.0):
        """Helper to scale and place mobjects on the defined grid."""
        mobject.scale(scale_factor)
        mobject.move_to(self.grid_positions[grid_pos])
        return mobject

    def construct(self):
        # Lecture points for the Duality of Cross Products
        lecture_data = [
            "- Cross Product as Area",
            "- The Unit Normal Vector",
            "- Linear Transformation View",
            "- The Dot Product Relationship",
            "- Geometric to Algebraic Duality"
        ]
        
        # Setup the scene layout
        self.setup_layout("The Duality of Cross Products", lecture_data)
        
        # Visual 1: Two vectors and their area
        vec_v = Arrow(ORIGIN, RIGHT * 1.5, color=BLUE, buff=0)
        vec_w = Arrow(ORIGIN, UP * 1.5, color=RED, buff=0)
        area_square = Square(side_length=1.5, fill_opacity=0.4, color=YELLOW, stroke_width=0)
        area_square.move_to(RIGHT * 0.75 + UP * 0.75)
        
        visual_group = VGroup(area_square, vec_v, vec_w)
        self.place_at_grid(visual_group, "B2", scale_factor=0.8)
        
        # Visual 2: A normal vector (Cross Product result)
        normal_vec = Arrow(ORIGIN, OUT * 1.5, color=GOLD, buff=0).rotate(45*DEGREES, axis=RIGHT)
        self.place_at_grid(normal_vec, "D2", scale_factor=1.0)

        # Animation Sequence
        self.play(Write(self.title_obj))
        self.play(FadeIn(self.lecture_group, shift=RIGHT))
        self.wait(1)

        # Show geometric interpretation
        self.play(Create(vec_v), Create(vec_w))
        self.play(FadeIn(area_square))
        self.wait(1)

        # Transform or label the cross product
        # Replaced MathTex with Text to avoid FileNotFoundError: 'latex'
        cp_label = Text("v × w", font_size=36, color=GOLD, slant=ITALIC)
        self.place_at_grid(cp_label, "A2")
        
        self.play(Write(cp_label))
        self.play(Create(normal_vec))
        
        # Final Transformation to show "Duality" concept
        target_circle = Circle(radius=0.6, color=WHITE, fill_opacity=0.2)
        self.place_at_grid(target_circle, "B2")
        
        self.play(
            Transform(visual_group, target_circle),
            FadeOut(normal_vec),
            run_time=2
        )
        
        self.wait(2)
