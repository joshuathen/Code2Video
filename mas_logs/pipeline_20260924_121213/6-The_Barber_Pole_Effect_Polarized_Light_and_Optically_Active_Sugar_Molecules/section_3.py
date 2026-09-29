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
            "Different colors rotate at different angles.",
            "Crossed polarizers reveal these color shifts.",
            "This creates the barber pole effect."
        ]
        self.setup_layout("The Barber Pole Effect Mechanism", lecture_lines)
        
        # Colors for visualization
        color_green = "#00FF00"

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(color_green)
        
        # Load asset
        barberpole_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/barberpole.svg")
        
        # Display light rays with varying rotations
        light_ray = VGroup(
            Line(ORIGIN, RIGHT*2, color="#FF5555", stroke_width=4),
            Line(ORIGIN, RIGHT*2, color="#55FF55", stroke_width=4),
            Line(ORIGIN, RIGHT*2, color="#5555FF", stroke_width=4)
        )
        self.place_at_grid(light_ray, 'B2', scale_factor=0.8)
        self.add(light_ray)
        
        # Rotate rays
        pivot = light_ray[0].get_start()
        self.play(
            Rotate(light_ray[0], angle=PI/8, about_point=pivot),
            Rotate(light_ray[1], angle=PI/6, about_point=pivot),
            Rotate(light_ray[2], angle=PI/4, about_point=pivot),
            run_time=2
        )

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(color_green)
        
        # Create crossed polarizers
        pol1 = Rectangle(height=2, width=0.2, color="#AAAAAA", fill_opacity=0.5)
        pol2 = Rectangle(height=2, width=0.2, color="#AAAAAA", fill_opacity=0.5)
        pols = VGroup(pol1, pol2)
        self.place_in_area(pols, 'D2', 'D5', scale_factor=0.9)
        self.play(Create(pol1), Create(pol2))
        
        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(color_green)
        
        # Show spiral path (barber pole effect)
        path = ParametricFunction(
            lambda t: np.array([t*0.5, np.sin(t*2)*0.5, 0]),
            t_range=np.array([0, 4*PI]),
            color=WHITE
        )
        self.place_at_grid(path, 'E3', scale_factor=0.7)
        
        # Add label
        label = Text("Wave Function", font_size=18)
        label.next_to(path, DOWN)
        self.add(label)
        
        # Add barberpole icon
        self.place_at_grid(barberpole_asset, 'F4', scale_factor=0.5)
        
        self.play(Create(path), FadeIn(barberpole_asset), run_time=3)
        self.wait(1)
