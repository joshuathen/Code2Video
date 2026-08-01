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
            'Increasing mass ratio makes the reflection angle tiny.',
            "Collisions map out the circle's arc length.",
            'Reversing direction requires traversing the full arc.',
            'Total reflections equal Pi divided by the angle.',
            'This count exactly matches the digits of Pi.'
        ]
        self.setup_layout("Connecting to Pi: The Arc Length", lecture_lines)

        # Colors
        COLOR_CIRCLE = "#00FF00"  # Green
        COLOR_THETA = "#00FFFF"   # Cyan
        COLOR_FORMULA = "#FFFFFF" # White
        COLOR_GOLD = "#FFD700"    # Gold
        COLOR_HIGHLIGHT = YELLOW

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(COLOR_HIGHLIGHT)
        
        circle = Circle(radius=1.8, color=COLOR_CIRCLE)
        self.place_in_area(circle, "A2", "D5")
        
        # Define a small theta for visualization
        theta_val = 0.4 
        start_angle = PI/2
        end_angle = start_angle + theta_val
        
        arc_theta = Arc(radius=1.8, start_angle=start_angle, angle=theta_val, color=COLOR_THETA, stroke_width=6)
        arc_theta.move_to(circle.get_center())
        
        # Label theta - Using Text to avoid LaTeX dependency
        theta_label = Text("θ", color=COLOR_THETA)
        self.place_at_grid(theta_label, "B5", scale_factor=0.8)
        
        # Formula theta approx sqrt(m/M) - Using Text to avoid LaTeX dependency
        theta_formula = Text("θ ≈ √(m/M)", color=COLOR_FORMULA)
        self.place_at_grid(theta_formula, "E2", scale_factor=0.7)
        
        self.play(Create(circle))
        self.play(Create(arc_theta), Write(theta_label))
        self.play(Write(theta_formula))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(COLOR_HIGHLIGHT)
        
        dot = Dot(circle.point_at_angle(start_angle), color=COLOR_THETA)
        self.add(dot)
        
        steps_count = 8
        step_angle = 0.2
        path_group = VGroup()
        
        for i in range(steps_count):
            new_angle = start_angle + (i + 1) * step_angle
            target_pos = circle.point_at_angle(new_angle)
            step_arc = Arc(radius=1.8, start_angle=start_angle + i * step_angle, angle=step_angle, color=COLOR_THETA, stroke_width=4)
            step_arc.move_to(circle.get_center())
            path_group.add(step_arc)
            self.play(
                dot.animate.move_to(target_pos),
                Create(step_arc),
                run_time=0.2
            )
            
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(COLOR_HIGHLIGHT)
        
        semi_circle = Arc(radius=1.8, start_angle=0, angle=PI, color=COLOR_GOLD, stroke_width=8)
        semi_circle.move_to(circle.get_center())
        
        pi_label = Text("π radians", color=COLOR_GOLD)
        self.place_at_grid(pi_label, "D3", scale_factor=0.8)
        
        self.play(Create(semi_circle), Write(pi_label))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(COLOR_HIGHLIGHT)
        
        # Formula N = pi / theta - Using Text to avoid LaTeX dependency
        n_formula = Text("N = π / θ", color=COLOR_FORMULA)
        self.place_at_grid(n_formula, "F2", scale_factor=0.8)
        
        calc_text = Text("If M = 100² m, θ ≈ 0.01", color=COLOR_FORMULA)
        self.place_in_area(calc_text, "E4", "E6", scale_factor=0.5)
        
        self.play(Write(n_formula))
        self.play(Write(calc_text))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(COLOR_HIGHLIGHT)
        
        # Final result N = 314 - Using VGroup of Text to maintain animation functionality
        result_parts = [
            Text("N = ", color=COLOR_FORMULA),
            Text("3", color=COLOR_FORMULA),
            Text("1", color=COLOR_FORMULA),
            Text("4", color=COLOR_FORMULA)
        ]
        result_text = VGroup(*result_parts).arrange(RIGHT, buff=0.1, aligned_edge=DOWN)
        self.place_in_area(result_text, "F4", "F6", scale_factor=0.8)
        
        self.play(Write(result_text[0]))
        self.wait(0.2)
        
        # Highlight digits in gold
        for i in range(1, 4):
            self.play(
                result_text[i].animate.set_color(COLOR_GOLD),
                result_text[i].animate.scale(1.2),
                run_time=0.5
            )
            
        self.wait(2)
