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
        lecture_lines = [
            "Plane angles dictate sphere behavior.",
            "Parabolas need only one tangent sphere.",
            "Hyperbolas place spheres in different nappes."
        ]
        self.setup_layout("Generalizing to Parabola and Hyperbola", lecture_lines)
        
        # Assets
        cone_img = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/cone.svg")
        sphere_img = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/sphere.svg")
        
        # Representations
        cone = Cone(base_radius=1, height=2, direction=DOWN).rotate(PI/2, axis=RIGHT)
        cone.set_fill(opacity=0.3).set_stroke(WHITE, width=1)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        # Fix 30: Adjust cone placement and scale
        self.place_in_area(cone, 'B3', 'E6', scale_factor=0.8)
        self.add(cone)
        # Asset usage
        self.place_at_grid(cone_img, "B2", scale_factor=0.5)
        self.play(FadeIn(cone_img))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#32CD32")
        parabola_path = ParametricFunction(
            lambda t: np.array([t, 0.5 * t**2 - 0.5, 0]), t_range=[-1, 1]
        ).set_color("#32CD32")
        # Fix 31: Adjust parabola placement
        self.place_at_grid(parabola_path, 'B4', scale_factor=0.6)
        self.play(Create(parabola_path))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#1E90FF")
        hyperbola_path = ParametricFunction(
            lambda t: np.array([t, 0.5 * t**2 + 0.5, 0]), t_range=[-1, 1]
        ).set_color("#1E90FF")
        hyperbola_path_2 = ParametricFunction(
            lambda t: np.array([t, -(0.5 * t**2 + 0.5), 0]), t_range=[-1, 1]
        ).set_color("#1E90FF")
        
        hyperbola_group = VGroup(hyperbola_path, hyperbola_path_2)
        # Fix 32: Use group for cleaner placement
        self.place_in_area(hyperbola_group, 'E4', 'F5', scale_factor=0.7)
        
        self.play(Transform(parabola_path, hyperbola_path))
        self.play(Create(hyperbola_path_2))
        # Asset usage
        self.place_at_grid(sphere_img, "F6", scale_factor=0.5)
        self.play(FadeIn(sphere_img))
        self.wait(2)
