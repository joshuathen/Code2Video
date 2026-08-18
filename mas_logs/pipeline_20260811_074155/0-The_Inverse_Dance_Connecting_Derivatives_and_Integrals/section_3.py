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
        self.setup_layout("The Fundamental Theorem: The Great Undo", [
            "Calculus has a secret: these operations are inverse partners.",
            "Differentiation breaks a function down into its rate.",
            "Integration acts like glue, rebuilding the original whole.",
            "Start with position, differentiate, then integrate to return home.",
            "This 'Great Undo' is the Fundamental Theorem of Calculus."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Calculus has a secret: these operations are inverse partners.
        self.play(self.lecture[0].animate.set_color(WHITE))
        
        # d/dx machine
        m1_box = RoundedRectangle(height=1.5, width=2, color="#A9A9A9", fill_opacity=0.5)
        m1_label = MathTex(r"\frac{d}{dx}", color=WHITE)
        machine1 = VGroup(m1_box, m1_label)
        self.place_in_area(machine1, "B2", "C3", scale_factor=0.8)
        
        # Integral machine
        m2_box = RoundedRectangle(height=1.5, width=2, color="#A9A9A9", fill_opacity=0.5)
        m2_label = MathTex(r"\int", color=WHITE)
        machine2 = VGroup(m2_box, m2_label)
        self.place_in_area(machine2, "D4", "E5", scale_factor=0.8)
        
        self.play(FadeIn(machine1), FadeIn(machine2))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Differentiation breaks a function down into its rate.
        self.play(self.lecture[1].animate.set_color("#800080"))
        
        func_in = MathTex("x^2", color=WHITE)
        self.place_at_grid(func_in, "A2", scale_factor=0.8)
        
        self.play(FadeIn(func_in))
        self.play(func_in.animate.move_to(machine1.get_center()), run_time=1)
        self.play(FadeOut(func_in, scale=0.5))
        
        func_out_1 = MathTex("2x", color="#800080")
        # Fix for Issue 28: Move output below the machine to avoid overlap
        self.place_at_grid(func_out_1, "D3", scale_factor=0.8)
        self.play(FadeIn(func_out_1, shift=DOWN))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Integration acts like glue, rebuilding the original whole.
        self.play(self.lecture[2].animate.set_color("#00FF00"))
        
        self.play(func_out_1.animate.move_to(machine2.get_center()), run_time=1)
        self.play(FadeOut(func_out_1, scale=0.5))
        
        func_out_2 = MathTex("x^2 + C", color="#00FF00")
        # Fix for Issue 29: Move output below the machine to avoid overlap
        self.place_at_grid(func_out_2, "F5", scale_factor=0.8)
        self.play(FadeIn(func_out_2, shift=DOWN))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Start with position, differentiate, then integrate to return home.
        self.play(self.lecture[3].animate.set_color(WHITE))
        
        # Visualizing the flow between machines
        arrow = Arrow(machine1.get_bottom(), machine2.get_top(), color=WHITE)
        self.play(Create(arrow))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # This 'Great Undo' is the Fundamental Theorem of Calculus.
        self.play(self.lecture[4].animate.set_color(WHITE))
        
        inv_ops = Text("Inverse Operations", color=WHITE, font_size=36)
        # Fix for Issue 30: Centering the label better within the math flow area
        self.place_in_area(inv_ops, "A4", "B5", scale_factor=0.7)
        
        self.play(Write(inv_ops))
        self.play(Flash(inv_ops, color=WHITE))
        self.wait(2)
