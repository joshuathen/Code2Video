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

class Section1Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Visualize a sphere in 3D space.",
            "Identify points directly opposite each other.",
            "These are called antipodal points.",
            "Example: North and South Poles.",
            "They share identical properties."
        ]
        self.setup_layout("Intuitive Prerequisite: Antipodal Points", lecture_lines)
        
        # Elements
        # Using SVG asset for sphere as requested
        sphere = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/sphere.svg", color=WHITE)
        
        dot_n = Dot(color=RED)
        dot_s = Dot(color=RED)
        label_n = Text("N", font_size=20, color=YELLOW)
        label_s = Text("S", font_size=20, color=YELLOW)
        label_antipodal = Text("Antipodal Points", font_size=20, color=RED)
        
        # Initial positioning
        scene_group = VGroup(sphere, dot_n, dot_s, label_n, label_s)
        self.place_in_area(scene_group, 'B4', 'E6', scale_factor=0.75)
        self.place_at_grid(label_n, 'C4', scale_factor=0.5)
        self.place_at_grid(label_s, 'F4', scale_factor=0.5)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(BLUE))
        self.play(FadeIn(sphere))
        label_sphere = Text("Sphere", font_size=20, color=WHITE).next_to(sphere, UP)
        self.play(Write(label_sphere))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(GREEN))
        self.play(FadeIn(dot_n), FadeIn(dot_s))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(ORANGE))
        self.place_at_grid(label_antipodal, 'A4', scale_factor=0.6)
        self.play(Write(label_antipodal))

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(YELLOW))
        self.play(Rotate(sphere, angle=PI/4, axis=RIGHT))
        
        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(RED))
        self.play(Indicate(dot_n, color=YELLOW), Indicate(dot_s, color=YELLOW))
        self.wait(1)
