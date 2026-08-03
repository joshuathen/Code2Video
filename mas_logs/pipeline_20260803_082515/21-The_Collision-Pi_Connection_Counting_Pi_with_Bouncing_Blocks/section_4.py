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
        # Setup title and lecture lines
        lecture_lines = [
            "Each collision maps to a reflection in phase space.",
            "Two reflections combine to form a single rotation.",
            "The rotation angle depends on the mass ratio.",
            "Every \"clack\" is a jump around the circle's perimeter.",
            "The total journey spans exactly pi radians."
        ]
        self.setup_layout("The Geometric Twist: Collisions as Rotations", lecture_lines)

        # Define the geometric elements
        center_pos = (self.grid["C3"] + self.grid["E5"]) / 2
        radius = 1.5
        circle = Circle(radius=radius, color=WHITE)
        circle.move_to(center_pos)
        
        # Axes for reference
        y_axis = Line(center_pos + UP * 2, center_pos + DOWN * 2, color=GRAY, stroke_width=1)
        x_axis = Line(center_pos + LEFT * 2, center_pos + RIGHT * 2, color=GRAY, stroke_width=1)
        
        # Assets
        wall_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/wall.svg").set_color(BLUE)
        block_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/block.svg").set_color(RED)
        
        # Position icons
        self.place_at_grid(wall_icon, "B2", scale_factor=0.5)
        self.place_at_grid(block_icon, "B4", scale_factor=0.5)

        # Reflection line 1: Y-axis (wall collision)
        line_wall = Line(center_pos + UP * 1.8, center_pos + DOWN * 1.8, color=BLUE)
        
        # Reflection line 2: Slanted line (block collision)
        # Using a larger angle for visual clarity: 20 degrees
        angle_alpha = 20 * DEGREES
        line_block = Line(
            center_pos + rotate_vector(UP * 1.8, angle_alpha),
            center_pos + rotate_vector(DOWN * 1.8, angle_alpha),
            color=RED
        )

        # State point P
        start_angle = -30 * DEGREES
        p_val = ValueTracker(start_angle)
        point_p = Dot(color=YELLOW)
        point_p.add_updater(lambda m: m.move_to(
            center_pos + np.array([
                radius * np.cos(p_val.get_value()),
                radius * np.sin(p_val.get_value()),
                0
            ])
        ))

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        self.add(circle, y_axis, x_axis)
        self.play(Create(line_wall), FadeIn(point_p), FadeIn(wall_icon))
        
        # Reflect point P across Y-axis
        # Mirror angle across pi/2: new_angle = pi - old_angle
        current_angle = p_val.get_value()
        target_angle_1 = PI - current_angle
        self.play(p_val.animate.set_value(target_angle_1), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(YELLOW)
        self.play(Create(line_block), FadeIn(block_icon))
        
        # Reflect point P across line_block
        # Angle of line_block is alpha + PI/2
        # Reflection of angle phi across line with angle psi is 2*psi - phi
        line_angle = angle_alpha + PI/2
        target_angle_2 = 2 * line_angle - target_angle_1
        self.play(p_val.animate.set_value(target_angle_2), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(YELLOW)
        # Rotation angle is 2 * alpha
        theta_arc = Arc(
            radius=0.5,
            start_angle=start_angle,
            angle=target_angle_2 - start_angle,
            arc_center=center_pos,
            color="#FFA500"
        )
        theta_label = MathTex(r"\theta", color="#FFA500")
        # Fix issue 33: Place theta_label at B5
        self.place_at_grid(theta_label, "B5", scale_factor=0.7)
        
        self.play(Create(theta_arc), Write(theta_label))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(YELLOW)
        # Show a few more jumps
        delta_theta = target_angle_2 - start_angle
        
        for i in range(2):
            next_angle = p_val.get_value() + delta_theta
            self.play(p_val.animate.set_value(next_angle), run_time=1)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(YELLOW)
        
        # Total arc spanning Pi
        pi_arc = Arc(
            radius=radius + 0.2,
            start_angle=start_angle,
            angle=PI,
            arc_center=center_pos,
            color=GREEN
        )
        pi_label = MathTex(r"\pi \text{ radians}", color=GREEN)
        # Fix issue 32: Place pi_label in area F4-F5
        self.place_in_area(pi_label, "F4", "F5", scale_factor=0.7)
        
        self.play(Create(pi_arc), Write(pi_label))
        self.wait(2)
