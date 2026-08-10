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
            "Collisions trace a path in phase space.",
            "The path reflects inside an angular wedge.",
            "High mass ratio creates a circular arc.",
            "Path length approximates the circle's circumference.",
            "This geometric arc reveals digits of Pi."
        ]
        self.setup_layout("Connecting Collisions to Pi", lecture_lines)
        
        # Define colors for lines
        colors = [BLUE, GREEN, YELLOW, ORANGE, RED]
        
        # Load Assets
        wedge_img = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/wedge.svg")
        arc_img = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/arc.svg")
        circle_img = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/circle.svg")

        # === Animation for Lecture Line 1 ===
        path = VGroup(
            Line(ORIGIN, RIGHT+UP), Line(RIGHT+UP, 2*RIGHT), Line(2*RIGHT, 2*RIGHT+DOWN)
        ).set_color(colors[0])
        self.place_in_area(path, 'B2', 'D5', scale_factor=0.6)
        self.play(Create(path))
        self.lecture[0].set_color(colors[0])
        
        # === Animation for Lecture Line 2 ===
        # Use Asset: wedge.svg
        self.place_in_area(wedge_img, 'B2', 'C4', scale_factor=0.7)
        self.play(FadeIn(wedge_img), path.animate.set_color(colors[1]))
        self.lecture[1].set_color(colors[1])
        
        # === Animation for Lecture Line 3 ===
        # Use Asset: arc.svg
        self.place_in_area(arc_img, 'B4', 'C6', scale_factor=0.7)
        self.play(ReplacementTransform(path, arc_img))
        self.lecture[2].set_color(colors[2])
        
        # === Animation for Lecture Line 4 ===
        # Use Asset: circle.svg and highlighted points
        dot = Dot(color=colors[3])
        self.place_at_grid(dot, 'E2', scale_factor=1.0)
        path_label = Text("Circumference", font_size=18, color=colors[3])
        self.place_at_grid(path_label, 'E3', scale_factor=1.0)
        self.play(FadeIn(dot), Write(path_label), FadeIn(circle_img.scale(0.3).move_to(self.grid['D3'])))
        self.lecture[3].set_color(colors[3])
        
        # === Animation for Lecture Line 5 ===
        pi_label = MathTex(r"\\pi \\approx 3.1415...", color=colors[4])
        self.place_at_grid(pi_label, 'F4', scale_factor=1.2)
        self.play(Write(pi_label))
        self.lecture[4].set_color(colors[4])
        self.wait(2)
