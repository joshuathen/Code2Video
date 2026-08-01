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
        # Setup the layout with title and lecture lines
        self.setup_layout(
            "Prerequisite: Defining the 'Jordan Curve'", 
            [
                "A Jordan curve is a simple, non-self-intersecting loop.", 
                "It can be wobbly like a rubber band.", 
                "However, it can never cross over itself like this."
            ]
        )
        
        # === Animation for Lecture Line 1 ===
        # Lecture line color matches animation
        self.lecture[0].set_color(WHITE)
        
        # Visual: White circle and label
        circle = Circle(radius=1.2, color=WHITE)
        self.place_in_area(circle, "B2", "C5")
        
        label_jordan = Text("Jordan Curve", font_size=20, color=WHITE)
        # Issue 42: align with center of shape below (columns 2-5)
        self.place_in_area(label_jordan, "A2", "A5", scale_factor=0.8)
        
        self.play(
            Create(circle),
            FadeIn(label_jordan)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(WHITE)
        
        # Visual: Deformation to wobbly shape (still non-self-intersecting)
        wobbly_curve = ParametricFunction(
            lambda t: (1.2 + 0.25 * np.sin(5 * t)) * np.array([np.cos(t), np.sin(t), 0]),
            t_range=[0, TAU],
            color=WHITE
        )
        # Using circle center to maintain position
        wobbly_curve.move_to(circle.get_center())
        
        self.play(
            Transform(circle, wobbly_curve),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Cross over example - Not a Jordan curve (color Red)
        self.lecture[2].set_color(RED)
        
        # Visual: Red figure-eight
        # Issue 41: reduce scale factor to 1.1 to avoid crowding/clipping
        figure_eight = ParametricFunction(
            lambda t: np.array([np.cos(t), np.sin(2 * t) / 2, 0]),
            t_range=[0, TAU],
            color=RED
        )
        self.place_in_area(figure_eight, "E2", "F5", scale_factor=1.1)
        
        # Red X mark
        cross_line1 = Line(UL, DR, color=RED).scale(0.5)
        cross_line2 = Line(UR, DL, color=RED).scale(0.5)
        cross = VGroup(cross_line1, cross_line2)
        # Align cross with the center of the area used for figure_eight
        self.place_in_area(cross, "E2", "F5", scale_factor=0.5)
        
        label_not_jordan = Text("NOT a Jordan Curve", font_size=20, color=RED)
        # Issue 40: use place_in_area for multi-word label centering
        self.place_in_area(label_not_jordan, "D2", "D5", scale_factor=0.8)
        
        self.play(
            Create(figure_eight),
            FadeIn(label_not_jordan)
        )
        self.play(
            Create(cross)
        )
        self.wait(2)
