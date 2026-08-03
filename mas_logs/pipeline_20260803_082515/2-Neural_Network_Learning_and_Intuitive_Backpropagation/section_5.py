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

class Section5Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "The loss landscape is like a hilly, foggy terrain.",
            "The gradient points in the steepest uphill direction.",
            "We take a small step in the opposite direction.",
            "This process iteratively moves the network toward minimum error.",
            "Walking downhill reduces the overall cost of our model."
        ]
        self.setup_layout("Gradient Descent: Walking Down the Hill", lecture_lines)

        # === Animation for Lecture Line 1 ===
        # Draw a smooth U-shaped curve in #FFFFFF representing the 'Loss Landscape'.
        self.lecture[0].set_color(YELLOW)
        # Issue 31: self.place_in_area(curve, 'B2', 'F6', scale_factor=0.8)
        curve = FunctionGraph(lambda x: 0.5 * (x**2), x_range=[-2.2, 2.2], color=WHITE)
        self.place_in_area(curve, 'B2', 'F6', scale_factor=0.8)
        
        self.play(Create(curve))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # The gradient points in the steepest uphill direction.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/robot.svg]
        robot = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/robot.svg")
        robot.scale(0.3)
        
        # Position logic relative to curve
        x_val = -1.8
        scale_f = 0.8
        v_offset = curve.get_bottom()
        # Point (x, 0.5x^2) relative to vertex, scaled
        robot_pos = v_offset + x_val * RIGHT * scale_f + (0.5 * x_val**2) * UP * scale_f
        robot.move_to(robot_pos)
        
        self.play(FadeIn(robot))
        
        # Gradient arrow (uphill)
        uphill_vec = np.array([-0.5, 0.9, 0])
        uphill_arrow = Arrow(
            start=robot.get_center(),
            end=robot.get_center() + uphill_vec,
            color="#00FF00",
            buff=0
        )
        
        # Issue 33: self.place_at_grid(gradient_label, 'B3', scale_factor=0.6)
        gradient_label = Text("Gradient", font_size=18, color="#00FF00")
        self.place_at_grid(gradient_label, 'B3', scale_factor=0.6)
        
        self.play(Create(uphill_arrow), FadeIn(gradient_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # We take a small step in the opposite direction.
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        downhill_vec = -uphill_vec
        downhill_arrow = Arrow(
            start=robot.get_center(),
            end=robot.get_center() + downhill_vec,
            color="#00FF00",
            buff=0
        )
        step_label = Text("Step Down", font_size=18, color="#00FF00")
        self.place_at_grid(step_label, 'C3', scale_factor=0.6)
        
        self.play(
            ReplacementTransform(uphill_arrow, downhill_arrow),
            FadeOut(gradient_label),
            FadeIn(step_label)
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # This process iteratively moves the network toward minimum error.
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        
        x_tracker = ValueTracker(-1.8)
        robot.add_updater(lambda m: m.move_to(
            v_offset + x_tracker.get_value() * RIGHT * scale_f + (0.5 * x_tracker.get_value()**2) * UP * scale_f
        ))
        
        self.play(
            x_tracker.animate.set_value(-1.0),
            FadeOut(downhill_arrow),
            FadeOut(step_label),
            run_time=1
        )
        
        self.play(x_tracker.animate.set_value(0), run_time=1.5)
        robot.clear_updaters()
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Walking downhill reduces the overall cost of our model.
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        
        min_error_text = Text("Minimum Error", font_size=20, color="#00FF00")
        # Issue 32: self.place_at_grid(min_error_text, 'F4', scale_factor=0.7)
        self.place_at_grid(min_error_text, 'F4', scale_factor=0.7)
        
        self.play(FadeIn(min_error_text))
        self.wait(2)
        
        # Cleanup
        self.lecture[4].set_color(WHITE)
        self.wait(2)
