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
        # 1. SETUP LAYOUT
        title_text = "Step-by-Step Application: The Rare Dragon Test"
        lecture_lines = [
            "Consider a rare dragon and a 99% accurate detector.",
            "Only one in a thousand creatures is a dragon.",
            "False alarms from non-dragons can outweigh true results.",
            "We calculate the total probability of a positive test.",
            "Bayes shows the real chance is only nine percent."
        ]
        self.setup_layout(title_text, lecture_lines)
        
        # Constants for colors
        COLOR_PRIOR = "#FFD700"      # Gold for P(Dragon)
        COLOR_LIKELIHOOD = "#ADD8E6" # Light Blue for P(+|Dragon)
        COLOR_FP = "#FFB6C1"         # Light Pink for P(+|No Dragon)
        COLOR_RESULT = "#00FF00"     # Green for final probability
        COLOR_NEUTRAL = "#FFFFFF"    # White for calculations
        
        # === Animation for Lecture Line 1 ===
        # L0: Consider a rare dragon and a 99% accurate detector.
        # Storyboard Anim 1: Display 'P(Dragon) = 0.001' (#FFD700) and 'P(No Dragon) = 0.999' (#FFFFFF).
        self.play(self.lecture[0].animate.set_color(COLOR_PRIOR))
        
        p_drag = MathTex(r"P(\text{Dragon}) = 0.001", color=COLOR_PRIOR, font_size=28)
        p_no_drag = MathTex(r"P(\text{No Dragon}) = 0.999", color=COLOR_NEUTRAL, font_size=28)
        
        self.place_at_grid(p_drag, "B2")
        self.place_at_grid(p_no_drag, "B5")
        
        self.play(Write(p_drag), Write(p_no_drag))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # L1: Only one in a thousand creatures is a dragon.
        # Storyboard Anim 2: Show 'P(+|Dragon) = 0.99' (#ADD8E6) and 'P(+|No Dragon) = 0.01' (#FFB6C1).
        self.play(self.lecture[1].animate.set_color(COLOR_LIKELIHOOD))
        
        p_pos_drag = MathTex(r"P(+|\text{Dragon}) = 0.99", color=COLOR_LIKELIHOOD, font_size=28)
        p_pos_no_drag = MathTex(r"P(+|\text{No Dragon}) = 0.01", color=COLOR_FP, font_size=28)
        
        self.place_at_grid(p_pos_drag, "C2")
        self.place_at_grid(p_pos_no_drag, "C5")
        
        self.play(Write(p_pos_drag), Write(p_pos_no_drag))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # L2: False alarms from non-dragons can outweigh true results.
        # Storyboard Anim 3: Flash the text 'If Alarm Sounds (+)...' (#FFFFFF) at the top.
        # Issue 27: Position at A1-A6.
        self.play(self.lecture[2].animate.set_color(COLOR_NEUTRAL))
        
        event_text = Text("If Alarm Sounds (+)...", color=COLOR_NEUTRAL, font_size=24)
        self.place_in_area(event_text, "A1", "A6", scale_factor=0.8)
        
        self.play(FadeIn(event_text))
        self.play(Indicate(event_text, color=COLOR_NEUTRAL))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # L3: We calculate the total probability of a positive test.
        # Storyboard Anim 4: Show calculation: P(+) = (0.99 * 0.001) + (0.01 * 0.999) approx 0.011 (#FFFFFF).
        # Issue 28: Position at D1-D6.
        self.play(self.lecture[3].animate.set_color(COLOR_NEUTRAL))
        
        p_plus_calc = MathTex(
            r"P(+) = (0.99 \times 0.001) + (0.01 \times 0.999)",
            color=COLOR_NEUTRAL, font_size=24
        )
        p_plus_res = MathTex(
            r"P(+) \approx 0.011",
            color=COLOR_NEUTRAL, font_size=28
        )
        
        self.place_in_area(p_plus_calc, "D1", "D6", scale_factor=0.7)
        self.place_at_grid(p_plus_res, "E3")
        
        self.play(Write(p_plus_calc))
        self.wait(0.5)
        self.play(Write(p_plus_res))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # L4: Bayes shows the real chance is only nine percent.
        # Storyboard Anim 5: Compute P(Dragon|+) = (0.99 * 0.001) / 0.011 approx 0.09, highlighted in green (#00FF00).
        # Issue 29: Position at F1-F6.
        self.play(self.lecture[4].animate.set_color(COLOR_RESULT))
        
        final_result = MathTex(
            r"P(\text{Dragon}|+) = \frac{0.99 \times 0.001}{0.011} \approx 0.09",
            color=COLOR_RESULT, font_size=30
        )
        
        self.place_in_area(final_result, "F1", "F6", scale_factor=0.7)
        
        self.play(Write(final_result))
        self.play(Circumscribe(final_result, color=COLOR_RESULT))
        self.wait(3)
