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

class Section4Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Robot position follows a specific function.",
            "Velocity is the derivative of position.",
            "Calculated derivative keeps the robot safe."
        ]
        self.setup_layout("Application: The Robot's Velocity", lecture_lines)
        
        # Paths for robot asset
        robot_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/robot.svg"
        
        # === Animation for Lecture Line 1 ===
        robot = SVGMobject(robot_path)
        self.place_at_grid(robot, 'C1', scale_factor=0.6)
        
        axes = Axes(x_range=[0, 4, 1], y_range=[0, 16, 4], axis_config={"include_tip": False})
        p_t = axes.plot(lambda t: 3 * t**2, color=WHITE)
        group = VGroup(axes, p_t)
        self.place_in_area(group, 'E1', 'F3', scale_factor=0.4)
        
        self.play(FadeIn(robot), Create(axes), Create(p_t))
        self.lecture[0].set_color(WHITE)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        t_line = axes.get_vertical_line(axes.c2p(3, 3*3**2), color=YELLOW)
        deriv = MathTex("p'(t) = 6t", color=GREEN)
        self.place_at_grid(deriv, 'C4', scale_factor=0.7)
        
        self.play(Create(t_line), Write(deriv))
        self.lecture[1].set_color(GREEN)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        collision_text = Text("Collision Avoided!", color="#00FFFF", font_size=24)
        self.place_at_grid(collision_text, 'B4', scale_factor=0.8)
        
        # Animate robot moving
        self.play(robot.animate.shift(RIGHT*1.5), Write(collision_text))
        self.lecture[2].set_color("#00FFFF")
        self.wait(2)
