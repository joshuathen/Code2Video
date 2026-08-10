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
            "Dimensions define how we measure space and volume.",
            "1D: Two points on a line.",
            "2D: A disk in a plane.",
            "3D: A volume in space.",
            "General formula: Sum of x squared equals r squared."
        ]
        self.setup_layout("Prerequisite Intuition: Expanding Dimensions", lecture_lines)
        
        # Colors for lecture lines
        colors = [BLUE, GREEN, YELLOW, ORANGE, RED]
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(colors[0])
        dot = Dot(color=WHITE)
        self.place_at_grid(dot, 'B4', scale_factor=1.0)
        self.play(FadeIn(dot))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(colors[1])
        line = Line(start=LEFT, end=RIGHT, color=WHITE).scale(0.5)
        self.place_at_grid(line, 'C4')
        dots = VGroup(Dot(color=WHITE), Dot(color=WHITE)).arrange(RIGHT, buff=1.5)
        self.place_at_grid(dots, 'C4')
        self.play(Transform(dot, dots))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(colors[2])
        disk_svg = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/disk.svg")
        self.place_at_grid(disk_svg, 'C5', scale_factor=1.5)
        self.play(Transform(dots, disk_svg))

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(colors[3])
        sphere = Sphere(radius=0.8, color=WHITE, fill_opacity=0.3)
        self.place_at_grid(sphere, 'C5', scale_factor=0.8)
        self.play(Transform(disk_svg, sphere))

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(colors[4])
        formula = MathTex(r"x_1^2 + x_2^2 + \dots + x_n^2 = r^2", color=WHITE)
        self.place_in_area(formula, 'D4', 'F6', scale_factor=0.9)
        self.play(Write(formula))
        self.wait(2)
