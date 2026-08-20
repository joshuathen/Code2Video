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
        lecture_lines = ["Vectors are directed arrows with magnitude.", "Vector addition moves an end effector.", "Scalar multiplication stretches or flips vectors."]
        self.setup_layout("Prerequisite: The Geometry of Vectors", lecture_lines)
        
        # Assets: SVGMobject is required for .svg files, ImageMobject is for rasters (png/jpg)
        robot = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/robot.svg")
        arm = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/arm.svg")
        
        axes = Axes(x_range=[-1, 5], y_range=[-1, 5], axis_config={"include_tip": True}).scale(0.5)
        self.place_in_area(axes, 'C4', 'F5', scale_factor=0.9)
        self.add(axes)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        self.place_at_grid(robot, "B2", scale_factor=0.3)
        self.add(robot)
        
        v = Vector([3, 2], color="#FF5733")
        v.shift(axes.c2p(0, 0) - v.get_start())
        v_label = MathTex(r"\vec{v} = (3, 2)", color="#FF8C00")
        self.place_at_grid(v_label, "B5", scale_factor=0.7)
        self.play(Create(v), Write(v_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        u = Vector([1, 1], color=BLUE)
        u.shift(axes.c2p(0, 0) - u.get_start())
        v_offset = Vector([3, 2], color=WHITE)
        v_offset.shift(axes.c2p(1, 1) - v_offset.get_start())
        sum_vec = Vector([4, 3], color=GREEN)
        sum_vec.shift(axes.c2p(0, 0) - sum_vec.get_start())
        
        self.play(Create(u), Create(v_offset))
        self.play(Create(sum_vec))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        self.place_at_grid(arm, "E2", scale_factor=0.3)
        self.add(arm)
        
        c = ValueTracker(1.0)
        v_scaled = Vector([3, 2], color="#33FF57")
        v_scaled.shift(axes.c2p(0, 0) - v_scaled.get_start())
        v_scaled.add_updater(lambda m: m.become(Vector([3*c.get_value(), 2*c.get_value()], color="#33FF57").shift(axes.c2p(0,0))))
        
        self.play(FadeOut(u), FadeOut(v_offset), FadeOut(sum_vec), FadeOut(v))
        self.add(v_scaled)
        self.play(c.animate.set_value(1.5), run_time=2)
        self.play(c.animate.set_value(-0.5), run_time=2)
        self.wait(1)
