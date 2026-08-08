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
        lecture_lines = [
            "Slice a cone to reveal conic sections.",
            "Circles, ellipses, parabolas, and hyperbolas emerge.",
            "Dandelin Spheres explain these shapes perfectly.",
            "Visualize the flashlight beam on the wall.",
            "Discover the hidden geometric Rosetta Stone."
        ]
        self.setup_layout("Introduction: The Geometry of a Sliced Cone", lecture_lines)
        
        # Assets
        cone_img = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/cone.svg").set_color(WHITE)
        flashlight_img = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/flashlight.svg")
        
        # Other elements
        plane = Rectangle(width=2, height=1.5, color=BLUE)
        ellipse = Ellipse(width=0.8, height=0.4, color="#FF5733")
        label_cone = Text("Cone", font_size=20, color="#E0E0E0")
        label_ellipse = Text("Ellipse", font_size=20, color="#E0E0E0")
        
        # === Animation for Lecture Line 1 ===
        self.place_at_grid(cone_img, 'C3')
        self.play(FadeIn(cone_img))
        self.lecture[0].set_color(WHITE)
        
        # === Animation for Lecture Line 2 ===
        self.place_at_grid(plane, 'C4', scale_factor=1.0)
        self.play(FadeIn(plane))
        self.lecture[1].set_color(YELLOW)
        
        # === Animation for Lecture Line 3 ===
        self.place_at_grid(ellipse, 'C5', scale_factor=1.0)
        self.play(Create(ellipse))
        self.lecture[2].set_color("#FF5733")
        
        # === Animation for Lecture Line 4 ===
        self.play(Rotate(cone_img, angle=PI/8))
        self.lecture[3].set_color(BLUE)
        
        # === Animation for Lecture Line 5 ===
        self.place_at_grid(label_cone, 'B3', scale_factor=0.8)
        self.place_at_grid(label_ellipse, 'D5', scale_factor=0.8)
        self.place_at_grid(flashlight_img, 'F6', scale_factor=0.4)
        self.play(Write(label_cone), Write(label_ellipse), FadeIn(flashlight_img))
        self.lecture[4].set_color(GREEN)
        
        self.wait(2)
