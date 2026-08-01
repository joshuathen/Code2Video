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
        self.setup_layout(
            "The Rule of Squares (Born's Rule)",
            [
                "Probability comes from the square of amplitude magnitudes.",
                "Total probability must always equal exactly one.",
                "The state vector always sits on a unit circle.",
                "Vector rotation changes the likelihood of measurement outcomes.",
                "A larger projection means a higher probability."
            ]
        )

        # Positioning references
        circle_center = self.grid["D2"]
        # Prob bar positions - Issue 33, 34 fixes: labels at F5, F6; bars at E5, E6
        bar_x_alpha = self.grid["E5"][0]
        bar_x_beta = self.grid["E6"][0]
        bar_y_base = self.grid["F5"][1] + 0.5
        
        # 1. Circle and Axes
        unit_radius = 1.3
        unit_circle = Circle(radius=unit_radius, color="#D3D3D3")
        unit_circle.move_to(circle_center)
        
        y_axis = Arrow(circle_center, circle_center + [0, unit_radius + 0.3, 0], buff=0, color=WHITE)
        x_axis = Arrow(circle_center, circle_center + [unit_radius + 0.3, 0, 0], buff=0, color=WHITE)
        
        y_label = MathTex(r"|0\rangle", color="#FFFFFF", font_size=24).next_to(y_axis, UP, buff=0.1)
        x_label = MathTex(r"|1\rangle", color="#FFFFFF", font_size=24).next_to(x_axis, RIGHT, buff=0.1)
        
        # 2. Vector
        angle_tracker = ValueTracker(PI / 3) # Start at some angle
        
        vector = Arrow(circle_center, circle_center + [unit_radius, 0, 0], buff=0, color="#00FF00", stroke_width=4)
        def update_vector(v):
            ang = angle_tracker.get_value()
            # Angle 0 is |0> (Y-axis), PI/2 is |1> (X-axis)
            v.put_start_and_end_on(circle_center, circle_center + [unit_radius * np.sin(ang), unit_radius * np.cos(ang), 0])
        vector.add_updater(update_vector)
        
        # 3. Probability Bars
        alpha_bar = Rectangle(width=0.4, height=0.01, color="#FFFFFF", fill_opacity=0.8, stroke_width=1)
        alpha_bar.move_to([bar_x_alpha, bar_y_base, 0], aligned_edge=DOWN)
        
        alpha_label = MathTex(r"|\alpha|^2", color="#FFFFFF", font_size=24)
        # Issue 33: Move alpha_label to F5, scale 0.8
        self.place_at_grid(alpha_label, "F5", scale_factor=0.8)
        
        beta_bar = Rectangle(width=0.4, height=0.01, color="#00FFFF", fill_opacity=0.8, stroke_width=1)
        beta_bar.move_to([bar_x_beta, bar_y_base, 0], aligned_edge=DOWN)
        
        beta_label = MathTex(r"|\beta|^2", color="#00FFFF", font_size=24)
        # Issue 34: Move beta_label to F6, scale 0.8
        self.place_at_grid(beta_label, "F6", scale_factor=0.8)

        def update_alpha_bar(b):
            ang = angle_tracker.get_value()
            prob = np.cos(ang)**2
            b.stretch_to_fit_height(max(0.01, prob * 2.5), about_edge=DOWN)
            b.move_to([bar_x_alpha, bar_y_base, 0], aligned_edge=DOWN)
            
        def update_beta_bar(b):
            ang = angle_tracker.get_value()
            prob = np.sin(ang)**2
            b.stretch_to_fit_height(max(0.01, prob * 2.5), about_edge=DOWN)
            b.move_to([bar_x_beta, bar_y_base, 0], aligned_edge=DOWN)

        alpha_bar.add_updater(update_alpha_bar)
        beta_bar.add_updater(update_beta_bar)

        # 4. Equation
        equation = MathTex(r"|\alpha|^2 + |\beta|^2 = 1", color="#FFFFFF", font_size=36)
        # Issue 32: Move equation to B2-B4, scale 0.6
        self.place_in_area(equation, "B2", "B4", scale_factor=0.6)

        # 5. Projections
        proj_y = DashedLine(color=GRAY, stroke_width=2)
        proj_x = DashedLine(color=GRAY, stroke_width=2)
        
        def update_proj_y(m):
            ang = angle_tracker.get_value()
            tip = circle_center + [unit_radius * np.sin(ang), unit_radius * np.cos(ang), 0]
            m.put_start_and_end_on(tip, circle_center + [0, unit_radius * np.cos(ang), 0])
            
        def update_proj_x(m):
            ang = angle_tracker.get_value()
            tip = circle_center + [unit_radius * np.sin(ang), unit_radius * np.cos(ang), 0]
            m.put_start_and_end_on(tip, circle_center + [unit_radius * np.sin(ang), 0, 0])
            
        proj_y.add_updater(update_proj_y)
        proj_x.add_updater(update_proj_x)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        self.play(Create(unit_circle), Create(y_axis), Create(x_axis), Write(y_label), Write(x_label))
        self.play(Create(vector))
        self.play(Create(alpha_bar), Create(beta_bar), Write(alpha_label), Write(beta_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        self.play(Write(equation))
        self.play(Indicate(equation))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        self.play(unit_circle.animate.set_stroke(width=6, color=WHITE), run_time=0.5)
        self.play(unit_circle.animate.set_stroke(width=2, color="#D3D3D3"), run_time=0.5)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        # Rotation showing probabilities change
        self.play(angle_tracker.animate.set_value(PI/6), run_time=1.5)
        self.wait(0.5)
        self.play(angle_tracker.animate.set_value(PI/3), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        self.play(Create(proj_y), Create(proj_x))
        # Rotate back to |0> to see alpha grow
        self.play(angle_tracker.animate.set_value(0.0), run_time=2)
        self.wait(2)
        
        self.lecture[4].set_color(WHITE)
        self.wait(1)
