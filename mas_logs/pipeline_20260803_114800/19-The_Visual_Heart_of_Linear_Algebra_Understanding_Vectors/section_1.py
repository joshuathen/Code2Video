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
        self.setup_layout("Prerequisite: From Points to Directions", [
            "A point represents a static location in space.",
            "A vector is different; it represents movement.",
            "Think of it as a journey, not a destination."
        ])

        # === Animation for Lecture Line 1 ===
        # Show a #FFFFFF point at (0,0) labeled 'Location'.
        self.play(self.lecture[0].animate.set_color(WHITE))
        
        # Issue 36: Adjust grid_plane area and scale to avoid cluttering lecture notes.
        grid_plane = NumberPlane(
            x_range=[-1, 5, 1],
            y_range=[-1, 5, 1],
            x_length=5,
            y_length=5,
            background_line_style={"stroke_opacity": 0.3}
        )
        self.place_in_area(grid_plane, 'A2', 'F6', scale_factor=0.9)
        
        location_dot = Dot(color="#FFFFFF")
        self.place_at_grid(location_dot, 'E2')
        
        # Issue 34: Move 'Location' label to F1 to avoid overlap with vertical axis.
        location_label = Text("Location", font_size=20, color="#FFFFFF")
        self.place_at_grid(location_label, 'F1', scale_factor=0.8)
        
        self.play(Create(grid_plane))
        self.play(FadeIn(location_dot), Write(location_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Draw a #FFD700 vector arrow starting from the #FFFFFF point.
        self.play(self.lecture[1].animate.set_color("#FFD700"))
        
        # Destination label at A5 to avoid horizontal grid lines (Issue 35).
        destination_label = Text("Destination", font_size=20, color="#FFD700")
        self.place_at_grid(destination_label, 'A5', scale_factor=0.8)
        
        vector_arrow = Arrow(
            start=self.grid['E2'],
            end=self.grid['C5'],
            color="#FFD700",
            buff=0.1,
            stroke_width=6
        )
        
        self.play(Create(vector_arrow))
        self.play(Write(destination_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Animate a #00FF00 dot traveling along the #FFD700 vector.
        self.play(self.lecture[2].animate.set_color("#00FF00"))
        
        traveling_dot = Dot(color="#00FF00")
        traveling_dot.move_to(vector_arrow.get_start())
        
        self.play(FadeIn(traveling_dot))
        self.play(MoveAlongPath(traveling_dot, vector_arrow), run_time=2)
        self.play(Flash(traveling_dot, color="#00FF00"))
        self.wait(2)
