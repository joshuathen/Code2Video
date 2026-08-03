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

class Section7Scene(TeachingScene):
    def construct(self):
        title = "Summary: The Iterative Loop"
        lecture_lines = [
            "Learning is a loop of guessing and correcting.",
            "Each iteration makes the network more accurate.",
            "Repetition creates intelligence from simple adjustments."
        ]
        self.setup_layout(title, lecture_lines)

        # Colors
        color_forward = "#88C0D0"
        color_loss = "#BF616A"
        color_backprop = "#EBCB8B"
        color_descent = "#A3BE8C"
        color_arrow = "#D8DEE9"
        color_trained = "#00FF00"

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(color_forward)
        
        # Icons for Forward, Loss, Backprop, and Descent
        forward_icon = Text("Forward", font_size=20, color=color_forward)
        loss_icon = Text("Loss", font_size=20, color=color_loss)
        backprop_icon = Text("Backprop", font_size=20, color=color_backprop)
        descent_icon = Text("Descent", font_size=20, color=color_descent)

        self.place_at_grid(forward_icon, "B2")
        self.place_at_grid(loss_icon, "B5")
        self.place_at_grid(backprop_icon, "E5")
        self.place_at_grid(descent_icon, "E2")

        # Arrows for the loop
        arrow_f2l = Arrow(forward_icon.get_right(), loss_icon.get_left(), color=color_arrow, buff=0.1)
        arrow_l2b = Arrow(loss_icon.get_bottom(), backprop_icon.get_top(), color=color_arrow, buff=0.1)
        arrow_b2d = Arrow(backprop_icon.get_left(), descent_icon.get_right(), color=color_arrow, buff=0.1)
        arrow_d2f = Arrow(descent_icon.get_top(), forward_icon.get_bottom(), color=color_arrow, buff=0.1)

        self.play(
            Write(forward_icon),
            Write(loss_icon),
            Write(backprop_icon),
            Write(descent_icon),
            Create(arrow_f2l),
            Create(arrow_l2b),
            Create(arrow_b2d),
            Create(arrow_d2f),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(color_descent)
        
        # Fade out loop to show Archer
        self.play(
            FadeOut(forward_icon, loss_icon, backprop_icon, descent_icon),
            FadeOut(arrow_f2l, arrow_l2b, arrow_b2d, arrow_d2f),
            run_time=1
        )

        # Archer and target (Represented by shapes)
        # Resolved Issue 36: Move Archer to D2 and Target to D6 to avoid clutter
        archer = Triangle(color=WHITE).rotate(-PI/2)
        self.place_at_grid(archer, "D2", scale_factor=0.5)
        
        target = VGroup(
            Circle(radius=0.5, color=WHITE),
            Circle(radius=0.3, color=RED),
            Circle(radius=0.1, color=YELLOW)
        )
        self.place_at_grid(target, "D6", scale_factor=0.8)
        
        self.play(FadeIn(archer), FadeIn(target))

        # Shoot sequence of arrows
        # Arrow 1: Miss high
        arrow1 = Line(archer.get_right(), target.get_top() + UP*0.5, color=YELLOW).add_tip(tip_length=0.2)
        # Arrow 2: Closer
        arrow2 = Line(archer.get_right(), target.get_center() + UP*0.2, color=YELLOW).add_tip(tip_length=0.2)
        # Arrow 3: Bullseye
        arrow3 = Line(archer.get_right(), target.get_center(), color=color_trained).add_tip(tip_length=0.2)

        self.play(Create(arrow1))
        self.wait(0.5)
        self.play(FadeOut(arrow1))
        self.play(Create(arrow2))
        self.wait(0.5)
        self.play(FadeOut(arrow2))
        self.play(Create(arrow3))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(color_trained)
        
        # Resolved Issue 37: Move checkmark to C4 to avoid overlap
        checkmark = VGroup(
            Line(LEFT*0.2 + DOWN*0.2, ORIGIN, color=color_trained, stroke_width=8),
            Line(ORIGIN, RIGHT*0.4 + UP*0.5, color=color_trained, stroke_width=8)
        )
        self.place_at_grid(checkmark, "C4", scale_factor=1.0)
        
        # Resolved Issue 38: Move "Model Trained" text to B4
        trained_text = Text("Model Trained", font_size=24, color=color_trained)
        self.place_at_grid(trained_text, "B4", scale_factor=1.0)

        self.play(
            FadeIn(checkmark, shift=UP),
            Write(trained_text),
            run_time=1.5
        )
        self.wait(2)
