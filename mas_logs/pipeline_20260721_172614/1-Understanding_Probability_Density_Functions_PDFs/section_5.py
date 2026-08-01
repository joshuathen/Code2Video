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
        # Data from shared state
        title = "Application: The Robo-Delivery Drone"
        lines = [
            "Meet Bolt, a delivery robot with a battery PDF.",
            "Shading the curve from eighty to one hundred percent.",
            "High probability ensures Bolt completes the delivery successfully."
        ]
        
        self.setup_layout(title, lines)
        
        # Colors based on storyboard
        color_bolt = "#bdc3c7"   # Gray
        color_area = "#2ecc71"   # Green
        color_prob = "#f1c40f"   # Gold
        
        # === Animation for Lecture Line 1 ===
        # Create Bolt the Robot
        body = Square(side_length=0.8, color=color_bolt, fill_opacity=0.8)
        head = Square(side_length=0.35, color=color_bolt, fill_opacity=0.8).next_to(body, UP, buff=0.05)
        eye_l = Dot(radius=0.04, color=BLACK).move_to(head.get_center() + LEFT*0.08 + UP*0.05)
        eye_r = Dot(radius=0.04, color=BLACK).move_to(head.get_center() + RIGHT*0.08 + UP*0.05)
        wheel_l = Circle(radius=0.12, color=color_bolt, fill_opacity=1).move_to(body.get_bottom() + LEFT*0.25)
        wheel_r = Circle(radius=0.12, color=color_bolt, fill_opacity=1).move_to(body.get_bottom() + RIGHT*0.25)
        bolt_mobjects = VGroup(body, head, eye_l, eye_r, wheel_l, wheel_r)
        bolt_label = Text("Bolt", font_size=18, color=color_bolt).next_to(bolt_mobjects, DOWN, buff=0.1)
        bolt_group = VGroup(bolt_mobjects, bolt_label)
        
        # Fix for Issue 40: Move Bolt to bottom right to avoid congestion
        self.place_in_area(bolt_group, 'F4', 'F6', scale_factor=0.8)
        
        # PDF Curve axes
        axes = Axes(
            x_range=[60, 105, 10],
            y_range=[0, 0.12, 0.04],
            x_length=4.5,
            y_length=3.0,
            axis_config={"include_tip": False, "font_size": 14, "color": WHITE},
            tips=False
        ).add_coordinates()
        
        # Place axes in upper-right area
        self.place_in_area(axes, "A2", "D5", scale_factor=0.8)
        
        def pdf_func(x):
            # Normal distribution centered at 90 with sigma 6
            return 0.1 * np.exp(-0.5 * ((x - 90) / 6) ** 2)
        
        curve = axes.plot(pdf_func, x_range=[60, 105], color=color_bolt)
        curve_label = MathTex("f(x)", font_size=20, color=color_bolt).next_to(curve, UP, buff=0.1)
        
        self.play(
            self.lecture[0].animate.set_color(color_bolt),
            FadeIn(bolt_group),
            Create(axes),
            Create(curve),
            Write(curve_label)
        )
        self.wait(1)
        
        # === Animation for Lecture Line 2 ===
        # Shade area from 80 to 100
        area = axes.get_area(curve, x_range=[80, 100], color=color_area, opacity=0.5)
        area_label = MathTex("P(80 \\le X \\le 100)", font_size=18, color=color_area)
        
        # Fix for Issue 38: Relocate area_label to A4 for better visual connection
        self.place_at_grid(area_label, 'A4', scale_factor=0.8)
        
        self.play(
            self.lecture[1].animate.set_color(color_area),
            FadeIn(area),
            Write(area_label)
        )
        self.wait(1)
        
        # === Animation for Lecture Line 3 ===
        # Display 75% probability
        prob_text = Text("75%", font_size=36, color=color_prob)
        
        # Fix for Issue 39: Move prob_text to E1-E3 to avoid overlap with graph
        self.place_in_area(prob_text, 'E1', 'E3', scale_factor=1.0)
        
        self.play(
            self.lecture[2].animate.set_color(color_prob),
            Flash(prob_text, color=color_prob, flash_radius=0.5),
            FadeIn(prob_text),
            bolt_group.animate.move_to(self.grid["F5"]),
            run_time=2
        )
        self.wait(2)
