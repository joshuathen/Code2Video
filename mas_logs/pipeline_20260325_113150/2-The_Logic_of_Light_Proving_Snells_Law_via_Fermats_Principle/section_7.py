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

class Section7Scene(TeachingScene):
    def construct(self):
        # Mandatory layout setup with prescribed lecture lines
        lecture_lines = [
            "A straw in water appears bent at the surface.",
            "Light rays change direction when moving between different media.",
            "Snell's Law explains this common optical illusion."
        ]
        self.setup_layout("Real-World Application: The Disappearing Straw", lecture_lines)

        # Color palette
        COLOR_STRAW = "#FFFFFF"
        COLOR_RAY = "#F1C40F"
        COLOR_APPARENT = "#FF0000"

        # === Animation for Lecture Line 1 ===
        # Replacing missing assets with VMobjects to avoid OSError
        # Beaker represented by a rounded rectangle
        beaker = VGroup(
            RoundedRectangle(height=3.5, width=2.5, corner_radius=0.2, color=BLUE_B, fill_opacity=0.4),
            Line([-1.25, 1.75, 0], [1.25, 1.75, 0], color=WHITE, stroke_width=2) # Rim
        )
        self.place_in_area(beaker, 'B2', 'F5', scale_factor=1.0)
        
        # Straw represented by a thick line
        straw_img = Line(UP * 1.5, DOWN * 1.5, stroke_width=10, color=COLOR_STRAW)
        straw_group = Group(straw_img)
        self.place_in_area(straw_group, 'A4', 'E4', scale_factor=0.9)
        
        # Display glass and straw
        self.play(FadeIn(beaker), FadeIn(straw_group), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight lecture line 2
        self.play(self.lecture[1].animate.set_color(COLOR_RAY))
        
        # Ray elements anchored to grid C3-E5
        ray_incident = Line(self.grid["E4"], self.grid["C4"], color=COLOR_RAY)
        ray_refracted = Arrow(self.grid["C4"], self.grid["B5"], color=COLOR_RAY, buff=0)
        ray_elements = VGroup(ray_incident, ray_refracted)
        # Note: self.place_in_area moves the whole group. 
        # For precision, we ensure the group is created near its intended area first.
        self.place_in_area(ray_elements, 'C3', 'E5', scale_factor=0.8)
        
        self.play(Create(ray_elements), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Highlight lecture line 3
        self.play(self.lecture[2].animate.set_color(COLOR_APPARENT))
        
        # Apparent straw represented by a line
        straw_apparent = Line(UP * 1.5, DOWN * 1.5, stroke_width=10, color=COLOR_APPARENT)
        # Position and rotate to simulate the 'bent' visual effect
        self.place_in_area(straw_apparent, 'A4', 'D4', scale_factor=0.9)
        straw_apparent.rotate(10 * DEGREES) 
        
        # Animate the shift to the apparent (broken) position
        self.play(
            straw_group.animate.set_opacity(0.3),
            FadeIn(straw_apparent),
            run_time=1.5
        )
        self.wait(2)
