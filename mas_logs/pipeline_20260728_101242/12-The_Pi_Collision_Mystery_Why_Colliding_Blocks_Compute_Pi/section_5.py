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
        title = "The Pi Connection: Arc Length and Wrapping"
        lines = [
            "The mass ratio determines the angle of each jump.",
            "Massive blocks create tiny, equal angular steps.",
            "The total path wraps around the circle's perimeter.",
            "We are essentially counting steps to measure Pi.",
            "Physics and geometry merge into one elegant result."
        ]
        self.setup_layout(title, lines)

        # === Animation for Lecture Line 1 ===
        # Label a red arc angle theta #FF0000 on the circle.
        self.lecture[0].set_color("#FF0000")
        
        # Circle at B2-E5 area with scale 0.8 (Resolving Issue 31)
        circle = Circle(radius=1.8, color=BLUE_E)
        self.place_in_area(circle, "B2", "E5", scale_factor=0.8)
        circle_center = circle.get_center()
        circle_radius = 1.8 * 0.8
        
        theta_val = 45 * DEGREES
        arc = Arc(radius=circle_radius, start_angle=0, angle=theta_val, color="#FF0000")
        arc.move_to(circle_center)
        
        theta_label = MathTex(r"\theta", color="#FF0000")
        # Positioning label near the arc within 1 grid unit
        label_offset = 0.5 * np.array([np.cos(theta_val/2), np.sin(theta_val/2), 0])
        theta_label.move_to(arc.point_from_proportion(0.5) + label_offset)
        
        radius_line_1 = Line(circle_center, circle_center + circle_radius * RIGHT, color=WHITE, stroke_width=2)
        radius_line_2 = Line(circle_center, circle_center + circle_radius * np.array([np.cos(theta_val), np.sin(theta_val), 0]), color=WHITE, stroke_width=2)

        self.play(Create(circle))
        self.play(Create(radius_line_1), Create(radius_line_2), Create(arc), Write(theta_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Shrink the arc #FF0000 as mass M of the block [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/block.svg] increases.
        # Resolving Issue 20: Asset integration
        self.lecture[1].set_color("#FF0000")
        
        block = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/block.svg")
        self.place_at_grid(block, "B1", scale_factor=0.6)
        block_label = MathTex("M", color=WHITE).next_to(block, UP, buff=0.1)
        
        new_theta_val = 15 * DEGREES
        new_arc = Arc(radius=circle_radius, start_angle=0, angle=new_theta_val, color="#FF0000")
        new_arc.move_to(circle_center)
        
        new_radius_line_2 = Line(circle_center, circle_center + circle_radius * np.array([np.cos(new_theta_val), np.sin(new_theta_val), 0]), color=WHITE, stroke_width=2)
        
        new_label_offset = 0.4 * np.array([np.cos(new_theta_val/2), np.sin(new_theta_val/2), 0])
        new_label_pos = new_arc.point_from_proportion(0.5) + new_label_offset
        
        self.play(FadeIn(block), Write(block_label))
        self.play(
            Transform(arc, new_arc),
            Transform(radius_line_2, new_radius_line_2),
            theta_label.animate.move_to(new_label_pos).scale(0.8),
            block.animate.scale(1.5), # Show mass increasing
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Fill the circle's perimeter with many small reflection steps.
        self.lecture[2].set_color("#00FF00")
        
        # We simulate the reflection steps on the circle
        num_steps = 15
        steps = VGroup()
        current_angle = 0
        step_angle = 24 * DEGREES # Arbitrary small angle for visualization
        for i in range(num_steps):
            p1 = circle_center + circle_radius * np.array([np.cos(current_angle), np.sin(current_angle), 0])
            current_angle += step_angle
            p2 = circle_center + circle_radius * np.array([np.cos(current_angle), np.sin(current_angle), 0])
            step_line = Line(p1, p2, color="#00FF00", stroke_width=3)
            steps.add(step_line)

        self.play(
            FadeOut(radius_line_1), 
            FadeOut(radius_line_2), 
            FadeOut(arc), 
            FadeOut(theta_label),
            FadeOut(block),
            FadeOut(block_label)
        )
        self.play(Create(steps, lag_ratio=0.2), run_time=3)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Display the formula N equals pi divided by theta in #FFFFFF.
        # Resolving Issue 30: Adjusting formula scale to 1.0
        self.lecture[3].set_color("#FFFFFF")
        
        formula = MathTex(r"N \approx \frac{\pi}{\theta}", color="#FFFFFF")
        self.place_in_area(formula, "A2", "A5", scale_factor=1.0)
        
        self.play(Write(formula))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Pulse the counter #FFFF00 as it reaches digits of Pi.
        self.lecture[4].set_color("#FFFF00")
        
        counter = Text("3.14159...", color="#FFFF00", font_size=36)
        self.place_in_area(counter, "F3", "F4", scale_factor=1.0)
        
        self.play(FadeIn(counter))
        self.play(Indicate(counter, color="#FFFF00", scale_factor=1.2))
        self.wait(2)
