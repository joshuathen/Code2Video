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

class Section2Scene(TeachingScene):
    def construct(self):
        self.setup_layout("The Conceptual Link: Accumulation Functions", 
                          ["Define an accumulation function: A(x).", 
                           "A(x) represents the area under f(t).", 
                           "A(x) is the integral from a to x."])
        
        # === Animation for Lecture Line 1 ===
        # Display accumulation function F(x) = ∫ f(t)dt
        eq = MathTex("A(x) = \\int_{a}^{x} f(t) dt", color="#FFFFFF")
        self.place_at_grid(eq, 'A3', scale_factor=1.0)
        self.play(Write(eq))
        self.lecture[0].set_color("#FFFFFF")

        # === Animation for Lecture Line 2 ===
        # Animate area under curve f(t) growing as x moves right
        axes = Axes(x_range=[0, 4, 1], y_range=[0, 3, 1], axis_config={"include_tip": False})
        curve = axes.plot(lambda t: 0.5 * (t - 2)**2 + 1, color=WHITE)
        x_tracker = ValueTracker(0.1)
        area = always_redraw(lambda: axes.get_area(curve, x_range=[0, x_tracker.get_value()], color="#00FF00", opacity=0.5))
        
        self.place_in_area(axes, 'B2', 'E5', scale_factor=0.7)
        self.add(axes, curve, area)
        self.play(x_tracker.animate.set_value(3), run_time=3)
        self.lecture[1].set_color("#00FF00")
        
        # === Animation for Lecture Line 3 ===
        # Label the instantaneous height of area as f(x)
        f_label = MathTex("f(x)", color="#FF0000")
        self.place_at_grid(f_label, 'D5', scale_factor=0.8)
        self.play(Write(f_label))
        self.lecture[2].set_color("#FF0000")
        
        self.wait(2)
