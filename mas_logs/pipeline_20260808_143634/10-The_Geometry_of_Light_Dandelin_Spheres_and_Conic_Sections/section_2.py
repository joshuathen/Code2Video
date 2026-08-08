from manim import *

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
        self.setup_layout("Introducing the Dandelin Spheres", [
            "A Dandelin sphere is placed inside the cone.",
            "It is tangent to both the cone and plane.",
            "This tangency point links geometry to the curve."
        ])
        
        # Load Assets
        cone = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/cone.svg")
        sphere = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/sphere.svg")
        plane = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/plane.svg")
        
        # Initialize positions
        self.place_at_grid(cone, 'D3', scale_factor=1.0)
        self.place_at_grid(sphere, 'D3', scale_factor=0.5)
        self.place_at_grid(plane, 'B3', scale_factor=0.8)
        
        tangent_point = Dot(color=RED).move_to(self.grid['C3'])

        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(cone), FadeIn(sphere))
        self.lecture[0].set_color("#00FFFF")
        self.play(sphere.animate.set_color("#00FFFF"))

        # === Animation for Lecture Line 2 ===
        self.play(FadeIn(plane))
        self.lecture[1].set_color("#FF00FF")
        self.play(plane.animate.set_color("#FF00FF"))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FF0000")
        self.play(FadeIn(tangent_point))
        self.wait(1)
