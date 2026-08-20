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

class Section5Scene(TeachingScene):
    def construct(self):
        lecture_lines = ["Duality simplifies complex network problems.",
                         "It aids circuit design and map coloring.",
                         "Use duality for efficient routing pathways."]
        self.setup_layout("Summary and Application", lecture_lines)

        # === Animation for Lecture Line 1: Euler's formula V - E + F = 2 ===
        formula = MathTex("V - E + F = 2", font_size=42, color=BLUE)
        self.place_in_area(formula, "B2", "B5", scale_factor=0.9)
        self.play(Write(formula))
        self.lecture[0].set_color(BLUE)
        self.wait(1)

        # === Animation for Lecture Line 2: Duality relationship V' = F, E' = E ===
        dual_text = MathTex("V' = F, \\quad E' = E", font_size=36, color=YELLOW)
        self.place_in_area(dual_text, "C2", "C5", scale_factor=0.8)
        self.play(FadeIn(dual_text))
        self.lecture[1].set_color(YELLOW)
        self.wait(1)

        # === Animation for Lecture Line 3: Map application ===
        # Using [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/map.svg]
        map_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/map.svg", color=GREEN)
        map_label = Text("Map Coloring", font_size=24, color=WHITE)
        map_label.scale(0.7)
        map_label.next_to(map_icon, DOWN)
        group = VGroup(map_icon, map_label)
        
        self.place_in_area(group, "E2", "F5", scale_factor=0.6)
        self.play(FadeIn(group))
        self.lecture[2].set_color(GREEN)
        self.wait(2)
