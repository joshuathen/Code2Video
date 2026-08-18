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
        self.setup_layout("Prerequisite: Fermat’s Principle of Least Time", [
            "Light seeks the path of least time.",
            "This is Fermat's Principle of least time.",
            "It governs light's behavior through media."
        ])

        # Assets
        light_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/light.svg")
        
        # Points and Labels
        Point_A = Dot(color=YELLOW)
        Point_B = Dot(color=YELLOW)
        self.place_at_grid(Point_A, 'B4', scale_factor=0.8)
        self.place_at_grid(Point_B, 'E5', scale_factor=0.8)
        A_label = Text("A", font_size=24).next_to(Point_A, UP)
        B_label = Text("B", font_size=24).next_to(Point_B, DOWN)
        
        # Paths
        path_a = Line(Point_A.get_center(), Point_B.get_center(), color=GREEN)
        path_b = CubicBezier(Point_A.get_center(), self.grid['C3'], self.grid['C5'], Point_B.get_center(), color=RED)
        Light_Path_Group = VGroup(path_a, path_b)
        self.place_in_area(Light_Path_Group, 'C3', 'E5', scale_factor=0.9)
        
        # Icon
        self.place_at_grid(light_icon, 'B3', scale_factor=0.5)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFD700")
        self.play(FadeIn(Point_A, A_label, Point_B, B_label, light_icon))
        self.play(Create(path_a), Create(path_b))
        self.wait(2)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color("#FFD700")
        self.play(Indicate(path_a))
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color("#FFD700")
        # Flash the path of least time
        highlight = light_icon.copy().set_color(WHITE).move_to(path_a.point_from_proportion(0.5))
        self.play(FadeIn(highlight), path_a.animate.set_color(WHITE))
        self.play(FadeOut(path_b), FadeOut(highlight))
        self.wait(2)
