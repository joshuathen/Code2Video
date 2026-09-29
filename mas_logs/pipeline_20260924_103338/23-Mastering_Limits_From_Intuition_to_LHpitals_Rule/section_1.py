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

class Section1Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Limits define a destination, not the point itself.",
            "Imagine a target zone for precision error.",
            "Epsilon is our allowed tolerance for output error.",
            "Delta controls input proximity to the target.",
            "Proving limits means guaranteeing output within epsilon."
        ]
        self.setup_layout("Intuitive Foundation & The Epsilon-Delta Challenge", lecture_lines)
        
        # Assets
        target_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/target.svg", color="#FFD700")
        anchor_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/anchor.svg", color="#32CD32")
        
        # Animation Elements
        curve = FunctionGraph(lambda x: 0.2 * x**2, x_range=[-2, 2], color="#FFFFFF")
        point = Dot(color="#FFD700")
        f_label = Text("f(x)", font_size=16, color="#FFD700")
        
        epsilon_box = Rectangle(width=0.8, height=0.8, color="#00BFFF")
        epsilon_label = Text("epsilon", font_size=16, color="#00BFFF")
        
        delta_seg = Line(start=ORIGIN, end=RIGHT*0.5, color="#FF4500")
        delta_label = Text("delta", font_size=16, color="#FF4500")
        limit_inequality = MathTex(r"|f(x) - L| < \epsilon", font_size=24, color="#FFFFFF")

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFD700"))
        self.play(Create(curve))
        self.place_at_grid(target_icon, 'E4', scale_factor=0.5)
        self.play(FadeIn(target_icon), FadeIn(point), Write(f_label.next_to(point, UP, buff=0.1)))
        self.play(point.animate.move_to(self.grid['E4']), run_time=2)
        
        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#00BFFF"))
        self.place_at_grid(epsilon_box, 'B5', scale_factor=0.7)
        self.play(Create(epsilon_box))
        
        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#00BFFF"))
        self.place_at_grid(epsilon_label, 'B4', scale_factor=0.8)
        self.play(Write(epsilon_label))
        
        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color("#FF4500"))
        self.place_at_grid(delta_seg, 'E5', scale_factor=0.7)
        self.place_at_grid(delta_label, 'E3', scale_factor=0.8)
        self.play(Create(delta_seg), Write(delta_label))
        
        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color("#FFFFFF"))
        self.place_at_grid(limit_inequality, 'D4', scale_factor=1.0)
        self.place_at_grid(anchor_icon, 'B2', scale_factor=0.5)
        self.play(Write(limit_inequality), FadeIn(anchor_icon))
