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
        title_text = "Conclusion: Sensitivity and Density"
        lecture_lines = [
            "Derivatives measure output sensitivity to input changes.",
            "High derivatives spread points out in the output.",
            "Low derivatives pack points closer together."
        ]
        
        self.setup_layout(title_text, lecture_lines)
        
        # Colors
        COLOR_SENSITIVITY = "#FFFF00"
        COLOR_INPUT = "#00FF00"
        COLOR_OUTPUT = "#00BFFF"
        
        # === Animation for Lecture Line 1 ===
        # Show word "Sensitivity" in #FFFF00
        self.lecture[0].set_color(COLOR_SENSITIVITY)
        
        sensitivity_word = Text("Sensitivity", color=COLOR_SENSITIVITY)
        # Resolved Issue 38: Use A2-A5 for sensitivity_word to avoid overlap with visual area
        self.place_in_area(sensitivity_word, 'A2', 'A5', scale_factor=0.9)
        
        self.play(Write(sensitivity_word))
        self.wait(1)
        
        # === Animation for Lecture Line 2 ===
        # High derivatives spread points out in the output.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(COLOR_OUTPUT)
        
        # Setup structural lines
        input_line = Line(self.grid["C2"], self.grid["C5"], color=COLOR_INPUT)
        output_line = Line(self.grid["E2"], self.grid["E5"], color=COLOR_OUTPUT)
        
        # Multi-word labels (L003)
        input_label = Text("Input Line", color=COLOR_INPUT)
        # Resolved Issue 39: Move input_label to B3-B4 to reduce crowding
        self.place_in_area(input_label, 'B3', 'B4', scale_factor=0.6)
        
        output_label = Text("Output Line", color=COLOR_OUTPUT)
        self.place_in_area(output_label, "D2", "D3", scale_factor=0.6)
        
        # High derivative scenario: Small input interval -> Large output interval
        in_dot1_h = Dot(self.grid["C3"], color=COLOR_INPUT)
        in_dot2_h = Dot(self.grid["C3"] + 0.3 * RIGHT, color=COLOR_INPUT)
        
        out_dot1_h = Dot(self.grid["E2"], color=COLOR_OUTPUT)
        out_dot2_h = Dot(self.grid["E5"], color=COLOR_OUTPUT)
        
        # Using Arrow to show mapping
        arrow1_h = Arrow(in_dot1_h.get_center(), out_dot1_h.get_center(), buff=0.1, color=GRAY, stroke_width=1)
        arrow2_h = Arrow(in_dot2_h.get_center(), out_dot2_h.get_center(), buff=0.1, color=GRAY, stroke_width=1)
        
        self.play(
            FadeOut(sensitivity_word),
            Create(input_line),
            Create(output_line),
            FadeIn(input_label),
            FadeIn(output_label)
        )
        
        self.play(FadeIn(in_dot1_h), FadeIn(in_dot2_h))
        self.play(GrowArrow(arrow1_h), GrowArrow(arrow2_h))
        self.play(FadeIn(out_dot1_h), FadeIn(out_dot2_h))
        self.wait(2)
        
        # === Animation for Lecture Line 3 ===
        # Low derivatives pack points closer together.
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(COLOR_OUTPUT)
        
        # Low derivative scenario: Large input interval -> Small output interval
        in_dot1_l = Dot(self.grid["C2"], color=COLOR_INPUT)
        in_dot2_l = Dot(self.grid["C5"], color=COLOR_INPUT)
        
        out_dot1_l = Dot(self.grid["E3"], color=COLOR_OUTPUT)
        out_dot2_l = Dot(self.grid["E3"] + 0.3 * RIGHT, color=COLOR_OUTPUT)
        
        arrow1_l = Arrow(in_dot1_l.get_center(), out_dot1_l.get_center(), buff=0.1, color=GRAY, stroke_width=1)
        arrow2_l = Arrow(in_dot2_l.get_center(), out_dot2_l.get_center(), buff=0.1, color=GRAY, stroke_width=1)
        
        # Transition out High derivative elements
        self.play(
            FadeOut(in_dot1_h), FadeOut(in_dot2_h), 
            FadeOut(out_dot1_h), FadeOut(out_dot2_h), 
            FadeOut(arrow1_h), FadeOut(arrow2_h)
        )
        
        self.play(FadeIn(in_dot1_l), FadeIn(in_dot2_l))
        self.play(GrowArrow(arrow1_l), GrowArrow(arrow2_l))
        self.play(FadeIn(out_dot1_l), FadeIn(out_dot2_l))
        self.wait(1)
        
        # Storyboard Step 4 (Closing Visual)
        # Final visual: Magnifying glass
        glass_circle = Circle(radius=0.15, color=WHITE)
        glass_handle = Line(ORIGIN, 0.15 * DOWN + 0.15 * RIGHT, color=WHITE).next_to(glass_circle, DOWN + RIGHT, buff=0)
        magnifying_glass = VGroup(glass_circle, glass_handle)
        
        # Resolved Issue 40: Scale magnifying_glass to 0.4 to prevent obstruction
        self.place_at_grid(magnifying_glass, 'C2', scale_factor=0.4)
        
        # Use persistent mobject + ValueTracker for movement (L010/L011)
        path_tracker = ValueTracker(0)
        magnifying_glass.add_updater(lambda m: m.move_to(
            interpolate(self.grid["C2"], self.grid["C5"], path_tracker.get_value())
        ))
        
        self.play(FadeIn(magnifying_glass))
        self.play(path_tracker.animate.set_value(1), run_time=2, rate_func=linear)
        magnifying_glass.remove_updater(magnifying_glass.updaters[0])
        self.play(FadeOut(magnifying_glass))
        self.wait(2)
