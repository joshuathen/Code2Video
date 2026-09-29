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
        self.setup_layout("Topological Intuition: The Winding Number", [
            "A loop wraps around the origin.",
            "The winding number detects trapped roots.",
            "No slip, then a root exists."
        ])

        # Objects
        origin = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/origin.svg")
        loop = ParametricFunction(lambda t: 1.0 * np.array([np.cos(t), np.sin(t), 0]), t_range=[0, 2*PI], color=BLUE)
        loop_label = Text("W-plane loop", font_size=18, color=BLUE)
        
        root = Dot(color=RED)
        root_label = Text("Root", font_size=18, color=RED)
        
        vector = Line(start=origin.get_center(), end=loop.point_from_proportion(0), color=LIGHT_GRAY)
        vector.add_updater(lambda mob: mob.put_start_and_end_on(origin.get_center(), loop.get_center() + np.array([1.0,0,0])))

        # === Animation for Lecture Line 1 ===
        self.place_at_grid(origin, 'C4', scale_factor=0.5)
        self.place_in_area(loop, 'B3', 'D5', scale_factor=0.7)
        self.place_at_grid(loop_label, 'A4', scale_factor=0.7)
        
        self.lecture[0].set_color("#FF6666")
        self.play(FadeIn(origin), Create(loop), Write(loop_label))

        # === Animation for Lecture Line 2 ===
        self.place_at_grid(root, 'C4', scale_factor=0.8)
        self.place_at_grid(root_label, 'E4', scale_factor=0.8)
        
        self.lecture[1].set_color("#77FF77")
        self.play(FadeIn(root), Write(root_label))
        
        # Simulate winding
        self.play(Rotate(loop, angle=2*PI, about_point=self.grid['C4'], run_time=3))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FF6666")
        
        # Highlight total winding count
        count_text = Text("Winding Number: 1", font_size=20, color=YELLOW)
        self.place_at_grid(count_text, 'F4')
        
        self.play(Write(count_text))
        self.wait(2)
