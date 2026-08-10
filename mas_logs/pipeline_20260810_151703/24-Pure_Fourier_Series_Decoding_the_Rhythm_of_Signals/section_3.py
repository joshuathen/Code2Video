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
        self.setup_layout("Visualization: The Epicycles Construction", [
            "Rotating vectors trace complex paths easily.",
            "Epicycles add layers to create shapes.",
            "Observe the Fourier robot drawing patterns.",
            "Circles define the rhythm of the motion.",
            "Geometry turns abstract math into clear visualization."
        ])

        # Assets
        robot_svg = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/robot.svg")
        
        # Setup epicycles
        center = self.grid["D4"]
        v1 = Vector([1.5, 0], color="#FF00FF")
        v2 = Vector([0.75, 0], color="#FFFF00")
        
        epicycle_group = VGroup(v1, v2)
        epicycle_group.move_to(center)
        
        circle1 = Circle(radius=1.5, color="#FF00FF", stroke_opacity=0.3).move_to(center)
        circle2 = Circle(radius=0.75, color="#FFFF00", stroke_opacity=0.3).move_to(center)

        self.place_at_grid(robot_svg, 'B4', scale_factor=0.4)
        
        # Path
        path = TracedPath(v2.get_end, stroke_color=WHITE, stroke_width=2)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FF00FF"), Create(circle1), GrowArrow(v1))
        
        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FFFF00"), Create(circle2), GrowArrow(v2), FadeIn(robot_svg))
        self.add(path)
        
        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#00FFFF"))
        v1_tracker = ValueTracker(0)
        v2_tracker = ValueTracker(0)
        
        v1.add_updater(lambda m: m.set_angle(v1_tracker.get_value()).shift(center - m.get_start()))
        v2.add_updater(lambda m: m.set_angle(v2_tracker.get_value()).shift(v1.get_end() - m.get_start()))
        circle2.add_updater(lambda m: m.move_to(v1.get_end()))
        
        self.play(v1_tracker.animate.set_value(2 * PI), v2_tracker.animate.set_value(4 * PI), run_time=4, rate_func=linear)
        
        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color("#00FF00"))
        
        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color("#FF0000"), FadeIn(robot_svg))
        
        self.wait(1)
