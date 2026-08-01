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
        # Data
        title_text = "The Adjustment: Turning the Knobs (Gradient Descent)"
        lecture_lines = [
            "We adjust weights using the calculated gradient direction.",
            "The learning rate sets our step size down.",
            "Optimal steps lead Nero straight to the bottom."
        ]
        
        self.setup_layout(title_text, lecture_lines)

        # === Animation for Lecture Line 1 ===
        # Description: Show a robotic hand turning the knob [Asset: /mmfs1/data/home/jthen/Code2Video/assets/icon/knob.svg] (#C0C0C0).
        self.lecture[0].set_color("#C0C0C0")
        
        # Load asset and apply fixes
        knob = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/knob.svg").set_color("#C0C0C0")
        self.place_at_grid(knob, "B4", scale_factor=0.8)
        
        # Simple robotic hand relative to knob
        hand_arm = Rectangle(width=0.7, height=0.15, color=GRAY_B, fill_opacity=1)
        hand_palm = Circle(radius=0.12, color=GRAY_B, fill_opacity=1)
        hand = VGroup(hand_arm, hand_palm).arrange(RIGHT, buff=0)
        hand.next_to(knob, RIGHT, buff=-0.1)
        
        self.play(FadeIn(knob), FadeIn(hand))
        # Turning animation
        self.play(
            Rotate(knob, angle=-PI/2, about_point=knob.get_center()),
            Rotate(hand, angle=-PI/2, about_point=knob.get_center()),
            run_time=1.5
        )
        self.wait(0.5)

        # === Animation for Lecture Line 2 ===
        # Description: Display the update formula W = W - ηG (#FFFFFF).
        self.lecture[1].set_color("#FFFFFF")
        
        # Formula W = W - ηG
        formula = Text("W = W - ηG", font_size=36, color="#FFFFFF")
        self.place_at_grid(formula, "D4", scale_factor=0.9)
        
        formula_label = Text("Update Rule", font_size=18, color=GRAY_A)
        formula_label.next_to(formula, UP, buff=0.2)
        
        self.play(Write(formula), FadeIn(formula_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Description: Animate a small step (#00FF00) moving down the error slope.
        self.lecture[2].set_color("#00FF00")
        
        # Error slope (simple parabola) confined to area E3-F5
        slope = FunctionGraph(
            lambda x: 0.25 * x**2 - 1, 
            x_range=[-2.0, 2.0], 
            color=WHITE
        )
        self.place_in_area(slope, "E3", "F5", scale_factor=0.8)
        
        # Positions on slope after placement
        start_pt = slope.point_from_proportion(0.15)
        mid_pt = slope.point_from_proportion(0.3)
        bottom_pt = slope.point_from_proportion(0.5)
        
        nero = Dot(start_pt, color=YELLOW, radius=0.1)
        nero_tag = Text("Nero", font_size=14, color=YELLOW).next_to(nero, UP, buff=0.1)
        
        self.play(Create(slope))
        self.play(FadeIn(nero), FadeIn(nero_tag))
        
        # Steps
        step_arrow_1 = Arrow(start_pt, mid_pt, color="#00FF00", buff=0, stroke_width=4, max_tip_length_to_length_ratio=0.3)
        self.play(GrowArrow(step_arrow_1))
        self.play(
            nero.animate.move_to(mid_pt),
            nero_tag.animate.next_to(mid_pt, UP, buff=0.1),
            run_time=0.8
        )
        
        step_arrow_2 = Arrow(mid_pt, bottom_pt, color="#00FF00", buff=0, stroke_width=4, max_tip_length_to_length_ratio=0.3)
        self.play(GrowArrow(step_arrow_2))
        self.play(
            nero.animate.move_to(bottom_pt),
            nero_tag.animate.next_to(bottom_pt, UP, buff=0.1),
            run_time=0.8
        )
        
        self.wait(2)
