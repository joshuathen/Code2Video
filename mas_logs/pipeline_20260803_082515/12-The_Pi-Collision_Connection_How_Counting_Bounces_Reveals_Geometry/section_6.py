from manim import *
import numpy as np

# Define colors based on storyboard and standard Manim colors
CYAN = TEAL
YELLOW = YELLOW
ORANGE = "#FF8C00"
PINK = "#FF69B4"
GREEN = "#00FF00"

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
        # Initial layout setup
        title = "The Final Link: Arc Length and Pi"
        lines = [
            "Each reflection covers a specific arc angle.",
            "The total bounces depend on the arc length.",
            "Larger mass ratios create smaller, more frequent steps.",
            "The number of steps perfectly calculates Pi's digits.",
            "Pure geometry reveals the secret of the bouncing blocks."
        ]
        self.setup_layout(title, lines)

        # Assets
        block_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/block.svg")
        model_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/model.svg")

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(CYAN)
        
        radius = 1.6
        semi_circle = Arc(radius=radius, start_angle=0, angle=PI, color=CYAN)
        # Issue 34: Reposition semi_circle to C2-F5
        self.place_in_area(semi_circle, 'C2', 'F5')
        
        # Calculate the actual arc origin
        # Note: Arc(0 to PI) bbox center is [0, r/2, 0]
        arc_origin = semi_circle.get_center() + DOWN * (radius / 2)
        base_line = Line(arc_origin + LEFT * radius, arc_origin + RIGHT * radius, color=GRAY)
        
        # Issue 35: Reposition label_pi to B3-B5
        label_pi = MathTex(r"\pi \text{ radians}", color=CYAN)
        self.place_in_area(label_pi, 'B3', 'B5', scale_factor=0.8)
        
        self.play(Create(semi_circle), Create(base_line), Write(label_pi))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        theta_val = PI / 6
        theta_arc = Arc(radius=0.6, start_angle=0, angle=theta_val, arc_center=arc_origin, color=YELLOW)
        theta_line_1 = Line(arc_origin, arc_origin + np.array([radius * np.cos(theta_val), radius * np.sin(theta_val), 0]), color=YELLOW)
        
        label_theta = MathTex(r"\theta", color=YELLOW)
        label_theta.move_to(arc_origin + 0.8 * (RIGHT * np.cos(theta_val/2) + UP * np.sin(theta_val/2)))
        
        # Asset: block.svg (Issue 27)
        self.place_at_grid(block_icon, 'D1', scale_factor=0.4)
        block_icon.set_color(YELLOW)
        
        self.play(Create(theta_arc), Create(theta_line_1), Write(label_theta), FadeIn(block_icon))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(ORANGE)
        
        # Animate multiple theta steps filling the semi-circle arc.
        num_steps = 5
        steps = VGroup()
        for i in range(1, num_steps + 1):
            angle = (i + 1) * theta_val
            if angle > PI: break
            step_line = Line(arc_origin, arc_origin + np.array([radius * np.cos(angle), radius * np.sin(angle), 0]), color=ORANGE, stroke_width=2)
            steps.add(step_line)
            
        self.play(Create(steps, lag_ratio=0.3))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(PINK)
        
        # Decrease theta size to show more steps for larger M.
        new_theta_val = PI / 15
        new_steps = VGroup()
        for i in range(1, 16):
            angle = i * new_theta_val
            step_line = Line(arc_origin, arc_origin + np.array([radius * np.cos(angle), radius * np.sin(angle), 0]), color=PINK, stroke_width=1)
            new_steps.add(step_line)
            
        self.play(
            FadeOut(steps),
            FadeOut(theta_line_1),
            FadeOut(theta_arc),
            FadeOut(label_theta),
            Create(new_steps, lag_ratio=0.1),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(GREEN)
        
        # Display final collision count matching Pi digits
        pi_digits = MathTex(r"N = 3, 31, 314, \dots", color=GREEN)
        self.place_at_grid(pi_digits, 'A5', scale_factor=1.2)
        
        # Asset: model.svg (Issue 27)
        self.place_at_grid(model_icon, 'E6', scale_factor=0.6)
        model_icon.set_color(GREEN)
        
        self.play(Write(pi_digits), FadeIn(model_icon))
        self.wait(3)
