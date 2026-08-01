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

class Section2Scene(TeachingScene):
    def construct(self):
        # Configuration
        YELLOW_C = "#FFFF00"
        CYAN_C = "#00FFFF" # standard cyan for better visibility
        PLANE_COLOR = "#00FF00"
        CONE_COLOR = WHITE
        ASSET_PATH = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/sphere.svg"
        
        self.setup_layout("The Magic Spheres Appear", [
            "Place two spheres inside the cone, above and below.",
            "Inflate them until they touch the cone's surface.",
            "Each sphere also touches the tilted cutting plane.",
            "They are tangent to the plane at points $F_1, F_2$.",
            "These tangent points are the foci of our ellipse."
        ])
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW_C)
        
        # Cone vertex at F3 (bottom center of right side)
        vertex = self.grid["F3"]
        w1_top = self.grid["A1"]
        w2_top = self.grid["A5"]
        wall1 = Line(vertex, w1_top, color=CONE_COLOR)
        wall2 = Line(vertex, w2_top, color=CONE_COLOR)
        
        # Plane line - crossing the cone diagonally
        p_start = self.grid["D1"]
        p_end = self.grid["B5"]
        plane_line = Line(p_start, p_end, color=PLANE_COLOR)
        
        # Initial small spheres using SVGMobject [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/sphere.svg]
        s1 = SVGMobject(ASSET_PATH).set_color(YELLOW_C).set_opacity(0.6).set_height(0.2)
        s2 = SVGMobject(ASSET_PATH).set_color(CYAN_C).set_opacity(0.6).set_height(0.2)
        
        self.place_at_grid(s1, "B2")
        self.place_at_grid(s2, "D4")
        
        self.play(Create(wall1), Create(wall2), Create(plane_line))
        self.play(FadeIn(s1), FadeIn(s2))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(CYAN_C)
        
        # Inflation targets (visually tangent to walls and plane)
        # Using set_height to scale the SVGMobject
        self.play(
            s1.animate.set_height(2.96).move_to(self.grid["B3"]),
            s2.animate.set_height(1.68).move_to(self.grid["D3"]),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(WHITE)
        
        # Highlight the contact points with the plane
        self.play(Indicate(s1, color=YELLOW_C), Indicate(s2, color=CYAN_C))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW_C)
        
        # Tangency points F1, F2 where spheres meet the plane line
        dot_f1 = Dot(color=YELLOW_C)
        dot_f2 = Dot(color=CYAN_C)
        self.place_at_grid(dot_f1, "B4")
        self.place_at_grid(dot_f2, "D2")
        
        label_f1 = MathTex("F_1", color=YELLOW_C, font_size=28)
        label_f2 = MathTex("F_2", color=CYAN_C, font_size=28)
        
        # Fixed positions for labels as per Issue 32 and 33
        self.place_at_grid(label_f1, "B5", scale_factor=0.8)
        self.place_at_grid(label_f2, "D1", scale_factor=0.8)
        
        self.play(
            FadeIn(dot_f1), FadeIn(dot_f2),
            Write(label_f1), Write(label_f2)
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(CYAN_C)
        
        # Final focus on the foci
        self.play(
            label_f1.animate.scale(1.2),
            label_f2.animate.scale(1.2),
            dot_f1.animate.scale(1.5),
            dot_f2.animate.scale(1.5)
        )
        self.play(
            label_f1.animate.scale(1/1.2),
            label_f2.animate.scale(1/1.2),
            dot_f1.animate.scale(1/1.5),
            dot_f2.animate.scale(1/1.5)
        )
        self.wait(2)
