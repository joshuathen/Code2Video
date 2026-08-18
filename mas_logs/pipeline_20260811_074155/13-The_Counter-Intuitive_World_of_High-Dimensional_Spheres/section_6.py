from manim import *
import numpy as np
import random

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

class Section6Scene(TeachingScene):
    def construct(self):
        # Initialize the layout
        self.setup_layout("Application: The Curse of Dimensionality", [
            "High-dimensional data points all become distant outliers.",
            "This makes clustering and finding centers extremely difficult.",
            "We call this the \"curse of dimensionality\" in science."
        ])

        COLOR_1 = "#00FFFF" # Cyan for Step 1
        COLOR_2 = "#FF00FF" # Magenta for Step 2
        COLOR_3 = "#FFFFFF" # White for Step 3

        # === Animation for Lecture Line 1 ===
        # High-dimensional data points all become distant outliers.
        self.play(self.lecture[0].animate.set_color(COLOR_1))
        
        # Create a cluster of points at the center
        num_points = 60
        # Use place_in_area logic to find center of the visual display zone
        center_pos = self.place_in_area(VMobject(), 'C3', 'D4').get_center()
        
        points = VGroup()
        for _ in range(num_points):
            offset = np.array([random.uniform(-0.3, 0.3), random.uniform(-0.3, 0.3), 0])
            dot = Dot(point=center_pos + offset, radius=0.04, color=COLOR_1)
            points.add(dot)
            
        self.play(Create(points))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # This makes clustering and finding centers extremely difficult.
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(COLOR_2)
        )
        
        # Define boundaries for the "edge" expansion (Area B2 to E5)
        tl = self.grid['B2']
        br = self.grid['E5']
        width = br[0] - tl[0]
        height = tl[1] - br[1]

        def get_perimeter_pos():
            side = random.randint(0, 3)
            if side == 0: # Top
                return tl + np.array([random.uniform(0, width), 0, 0])
            elif side == 1: # Bottom
                return br - np.array([random.uniform(0, width), 0, 0])
            elif side == 2: # Left
                return tl - np.array([0, random.uniform(0, height), 0])
            else: # Right
                return br + np.array([0, random.uniform(0, height), 0])

        # Animate points moving to the edges of the boundary box
        edge_animations = []
        for dot in points:
            target = get_perimeter_pos()
            edge_animations.append(dot.animate.move_to(target).set_color(COLOR_2))
            
        self.play(*edge_animations, run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # We call this the "curse of dimensionality" in science.
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(COLOR_3)
        )
        
        # Display a 'Data Robot' icon (Manual construction using primitives)
        robot_body = Square(side_length=0.4, color=COLOR_3, fill_opacity=1)
        robot_head = Circle(radius=0.15, color=COLOR_3, fill_opacity=1).next_to(robot_body, UP, buff=0.05)
        robot_eye_l = Dot(radius=0.03, color=BLACK).move_to(robot_head.get_center() + LEFT*0.06 + UP*0.02)
        robot_eye_r = Dot(radius=0.03, color=BLACK).move_to(robot_head.get_center() + RIGHT*0.06 + UP*0.02)
        robot_antenna = Line(robot_head.get_top(), robot_head.get_top() + UP*0.12, color=COLOR_3)
        
        robot = VGroup(robot_body, robot_head, robot_eye_l, robot_eye_r, robot_antenna)
        
        # Position robot in the center area
        self.place_in_area(robot, 'C3', 'D4')
        
        self.play(FadeIn(robot))
        # Search animation (scanning movement)
        self.play(robot.animate.rotate(0.2, about_point=robot.get_bottom()), run_time=0.4)
        self.play(robot.animate.rotate(-0.4, about_point=robot.get_bottom()), run_time=0.8)
        self.play(robot.animate.rotate(0.2, about_point=robot.get_bottom()), run_time=0.4)
        
        # Sad failure tilt
        self.play(robot.animate.rotate(0.25, axis=OUT), run_time=0.5)
        
        self.wait(3)
