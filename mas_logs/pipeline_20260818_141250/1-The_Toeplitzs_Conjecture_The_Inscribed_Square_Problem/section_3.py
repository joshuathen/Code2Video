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

class Section3Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Map pairs of points on the curve.",
            "These pairs represent chords in the configuration space.",
            "We transform geometry into a topological intersection problem."
        ]
        self.setup_layout("The Configuration Space Approach", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        # Display two points on a curve, connected by a line (chord)
        curve = Circle(radius=1, color=BLUE)
        p1 = Dot(curve.point_from_proportion(0.1), color="#FFA500")
        p2 = Dot(curve.point_from_proportion(0.4), color="#FFA500")
        chord = Line(p1.get_center(), p2.get_center(), color="#FFA500")
        
        group1 = VGroup(curve, p1, p2, chord)
        self.place_at_grid(group1, 'B2', scale_factor=0.7)
        self.play(Create(curve), Create(p1), Create(p2), Create(chord))
        self.lecture[0].set_color("#FFA500")

        # === Animation for Lecture Line 2 ===
        # Shift the chord representation to a point in a 2D plane (configuration space)
        config_space = Square(side_length=2, color=WHITE)
        point_in_space = Dot(config_space.get_center() + np.array([0.2, -0.3, 0]), color="#FFFFFF")
        
        group2 = VGroup(config_space, point_in_space)
        self.place_at_grid(group2, 'E4', scale_factor=0.6)
        
        self.play(Create(config_space), FadeIn(point_in_space))
        self.lecture[1].set_color("#FFFFFF")

        # === Animation for Lecture Line 3 ===
        # Show multiple intersection points in the configuration space
        intersections = VGroup(*[
            Dot(config_space.get_center() + np.array([0.5 * np.cos(t), 0.5 * np.sin(t), 0]), color="#FF0000")
            for t in np.linspace(0, 2*PI, 6)
        ])
        
        self.play(FadeIn(intersections))
        self.lecture[2].set_color("#FF0000")
        self.wait(2)
