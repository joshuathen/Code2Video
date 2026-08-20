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
        lecture_lines = [
            "Archimedes proved the sphere equals a circumscribed cylinder.",
            "This cylinder's surface area is four circles.",
            "The sphere's surface is four times its shadow.",
            "Mathematically, this equals four pi r squared.",
            "Geometry links the surface to the shadow."
        ]
        self.setup_layout("Connecting Area to Projection", lecture_lines)
        
        # Assets
        sphere = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/sphere.svg", color="#00FFFF")
        cylinder = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/cylinder.svg", color="#FFFFFF")
        
        # Positional fixes
        self.place_in_area(sphere, 'B2', 'B4', scale_factor=0.6)
        
        line = Line(start=LEFT, end=RIGHT, color=WHITE).scale(1.5)
        self.place_at_grid(line, 'D3', scale_factor=0.9)
        
        label = Text("Shadow", font_size=20, color=WHITE)
        self.place_at_grid(label, 'E3', scale_factor=0.8)

        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(sphere), self.lecture[0].animate.set_color("#00FFFF"))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FF00FF"))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FFFF00"))

        # === Animation for Lecture Line 4 ===
        self.play(FadeIn(line), FadeIn(label), self.lecture[3].animate.set_color("#00FF00"))

        # === Animation for Lecture Line 5 ===
        # Show cylinder appearing as part of the geometry
        self.play(FadeIn(cylinder), self.lecture[4].animate.set_color("#FF0000"))
        
        self.wait(2)
