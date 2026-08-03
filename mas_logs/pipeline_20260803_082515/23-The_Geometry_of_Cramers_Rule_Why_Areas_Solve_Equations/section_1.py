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
        # Setup the scene
        lecture_lines = [
            "Meet our robot needing exact fuel amounts.",
            "It must reach a target using two thrusters.",
            "This is a system of linear equations."
        ]
        self.setup_layout("Introduction: The Mixing Robot Mystery", lecture_lines)
        
        # Colors
        color_robot = WHITE
        color_target = "#00FF00"
        color_v1 = "#FF00FF"
        color_v2 = "#00FFFF"
        color_eq = WHITE
        color_highlight = "#FFFF00"

        # === Animation for Lecture Line 1 ===
        # Show a robot icon [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/robot.svg] at the origin in #FFFFFF.
        
        # Define the coordinate plane
        # Adjusted area to B1-F6 to avoid collision with equation (Issue 32)
        plane = NumberPlane(
            x_range=[-1, 9, 2],
            y_range=[-1, 13, 2],
            x_length=5,
            y_length=5,
            axis_config={"include_tip": True, "color": GREY},
            background_line_style={"stroke_opacity": 0.2}
        )
        self.place_in_area(plane, 'B1', 'F6')
        
        # Robot icon (Issue 31)
        robot = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/robot.svg").set_color(color_robot).scale(0.3)
        robot.move_to(plane.coords_to_point(0, 0))
        robot_label = Text("Robot", font_size=16, color=color_robot).next_to(robot, DL, buff=0.1)

        self.play(self.lecture[0].animate.set_color(color_robot))
        self.play(Create(plane), FadeIn(robot, robot_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Display target vector b = [7, 11] in #00FF00.
        # Display thruster vectors [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/thruster.svg] v1 = [2, 1] in #FF00FF and v2 = [1, 3] in #00FFFF.

        b_vec = Arrow(plane.coords_to_point(0, 0), plane.coords_to_point(7, 11), buff=0, color=color_target, stroke_width=4)
        b_label = MathTex("\\vec{b}", color=color_target).scale(0.8).next_to(b_vec.get_end(), UR, buff=0.1)
        
        # Thruster icons (Issue 31)
        thruster_v1 = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/thruster.svg").set_color(color_v1).scale(0.2)
        v1_vec = Arrow(plane.coords_to_point(0, 0), plane.coords_to_point(2, 1), buff=0, color=color_v1, stroke_width=4)
        v1_label = MathTex("\\vec{v}_1", color=color_v1).scale(0.8)
        v1_group = VGroup(thruster_v1, v1_label).arrange(RIGHT, buff=0.1).next_to(v1_vec.get_end(), RIGHT, buff=0.1)
        
        thruster_v2 = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/thruster.svg").set_color(color_v2).scale(0.2)
        v2_vec = Arrow(plane.coords_to_point(0, 0), plane.coords_to_point(1, 3), buff=0, color=color_v2, stroke_width=4)
        v2_label = MathTex("\\vec{v}_2", color=color_v2).scale(0.8)
        v2_group = VGroup(thruster_v2, v2_label).arrange(RIGHT, buff=0.1).next_to(v2_vec.get_end(), UP, buff=0.1)

        self.play(self.lecture[1].animate.set_color(color_target))
        self.play(GrowArrow(b_vec), Write(b_label))
        self.play(GrowArrow(v1_vec), FadeIn(v1_group), GrowArrow(v2_vec), FadeIn(v2_group))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Animate the equation x*v1 + y*v2 = b appearing in #FFFFFF.
        # Highlight the unknowns x and y with #FFFF00.
        
        # Adjusting equation placement and scale (Issue 33)
        eq = MathTex("x", "\\vec{v}_1", "+", "y", "\\vec{v}_2", "=", "\\vec{b}", color=color_eq)
        self.place_in_area(eq, 'A2', 'A5', scale_factor=1.0)
        
        self.play(self.lecture[2].animate.set_color(color_highlight))
        self.play(Write(eq))
        self.wait(0.5)
        
        # Highlight x and y
        self.play(
            eq[0].animate.set_color(color_highlight),
            eq[3].animate.set_color(color_highlight)
        )
        self.wait(2)
