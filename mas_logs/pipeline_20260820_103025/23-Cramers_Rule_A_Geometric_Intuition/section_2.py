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
        self.setup_layout("Framing the Linear System", [
            "Equations represent building a vector 'b' from v1, v2.",
            "Think of this like a robot arm reaching a target.",
            "Variables x1, x2 define our reach in each direction."
        ])
        
        # Setup robot icon
        robot_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/robot.svg")
        self.place_at_grid(robot_icon, 'A4', scale_factor=0.3)
        
        # Setup vectors
        axes = Axes(x_range=[-1, 4, 1], y_range=[-1, 3, 1], axis_config={"include_tip": False})
        v1 = Vector([1, 1], color=BLUE)
        v2 = Vector([2, -0.5], color=BLUE)
        v_b = Vector([3, 1.5], color="#FF4500") # b
        
        # Tethered labels
        l1 = MathTex("v_1", color=BLUE, font_size=24)
        l2 = MathTex("v_2", color=BLUE, font_size=24)
        lb = MathTex("b", color="#FF4500", font_size=24)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFD700")
        
        self.place_in_area(axes, 'B3', 'E5', scale_factor=0.6)
        
        # Align vectors to origin of axes
        v1.move_to(axes.c2p(0.5, 0.5))
        v2.move_to(axes.c2p(1, -0.25))
        
        l1.next_to(v1.get_end(), RIGHT, buff=0.1)
        l2.next_to(v2.get_end(), RIGHT, buff=0.1)
        
        self.play(Create(axes), Create(robot_icon), Create(v1), Write(l1), Create(v2), Write(l2))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#00FF00")
        
        v_b.move_to(axes.c2p(1.5, 0.75))
        self.place_at_grid(lb, 'C6', scale_factor=0.8)
        lb.next_to(v_b.get_end(), UP, buff=0.1)
        
        self.play(Create(v_b), Write(lb))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FFFFFF")
        
        formula = MathTex("x_1v_1 + x_2v_2 = b", font_size=24)
        self.place_in_area(formula, 'E2', 'F5', scale_factor=0.9)
        
        self.play(FadeIn(formula))
        self.wait(2)
