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
        self.setup_layout("The Lifeguard Problem", [
            "A lifeguard must reach a swimmer quickly.",
            "Running speed on sand exceeds swimming speed.",
            "The path bends to minimize total time."
        ])
        
        # Define elements using SVG assets
        beach = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/beach.svg", color="#FFD700")
        lifeguard = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/lifeguard.svg", color="#FFFFFF")
        swimmer = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/swimmer.svg", color="#00FFFF")
        
        boundary = Line(start=[-3, 0, 0], end=[3, 0, 0], color=WHITE)
        
        # Positioning based on grid feedback
        self.place_at_grid(beach, 'B2', scale_factor=0.6)
        self.place_at_grid(boundary, 'C2', scale_factor=0.6)
        self.place_at_grid(lifeguard, 'A1', scale_factor=0.5)
        self.place_at_grid(swimmer, 'F6', scale_factor=0.5)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFFFFF")
        self.play(FadeIn(beach), FadeIn(lifeguard), FadeIn(swimmer), Create(boundary))
        self.wait(2)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color("#FF0000")
        
        sand_label = Text("Faster", font_size=20, color="#FFFF00")
        water_label = Text("Slower", font_size=20, color="#0000FF")
        
        self.place_at_grid(sand_label, 'B4', scale_factor=0.8)
        self.place_at_grid(water_label, 'E4', scale_factor=0.8)
        
        self.play(Write(sand_label), Write(water_label))
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color("#FF4500")
        
        # Create bent path
        path = VMobject(color="#FF4500")
        path.set_points_smoothly([lifeguard.get_center(), [1.5, 0, 0], swimmer.get_center()])
        
        self.play(Create(path))
        self.wait(3)
