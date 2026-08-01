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

class Section6Scene(TeachingScene):
    def construct(self):
        # Mandatory layout call
        self.setup_layout("Summary & The 'Golden Rule'", [
            "Always treat y as a hidden function of x.",
            "Remember, the derivative of any constant is always zero.",
            "Follow the flow: Differentiate, then group, then solve."
        ])

        # === Animation for Lecture Line 1 ===
        # Highlight lecture line 1 in yellow
        self.play(self.lecture[0].animate.set_color("#FFFF00"))
        
        # Display 'The Golden Rule: Treat y as f(x)' in yellow (#FFFF00)
        golden_rule_text = Text("The Golden Rule:\nTreat y as f(x)", color="#FFFF00", font_size=28)
        self.place_in_area(golden_rule_text, "A2", "B5")
        self.play(Write(golden_rule_text))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight lecture line 2 in light red
        self.play(self.lecture[1].animate.set_color("#FFCCCC"))
        
        # Animate 'd/dx [constant] = 0' with the 0 flashing in red (#FF0000)
        constant_deriv = VGroup(Text("d/dx [ constant ] =", font_size=36), Text("0", font_size=36)).arrange(RIGHT)
        self.place_at_grid(constant_deriv, "C3")
        self.play(FadeIn(constant_deriv))
        
        # Zero flashes in red
        zero_mobj = constant_deriv[1]
        self.play(Indicate(zero_mobj, color="#FF0000", scale_factor=2.5))
        self.play(zero_mobj.animate.set_color("#FF0000"))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Highlight lecture line 3 in light green
        self.play(self.lecture[2].animate.set_color("#AAFFAA"))
        
        # Show flow chart: 'Differentiate' -> 'Group' -> 'Solve' in white boxes.
        step1_text = Text("Differentiate", font_size=20)
        step1_box = SurroundingRectangle(step1_text, color=WHITE, buff=0.15)
        step1 = VGroup(step1_box, step1_text)
        
        step2_text = Text("Group", font_size=20)
        step2_box = SurroundingRectangle(step2_text, color=WHITE, buff=0.15)
        step2 = VGroup(step2_box, step2_text)
        
        step3_text = Text("Solve", font_size=20)
        step3_box = SurroundingRectangle(step3_text, color=WHITE, buff=0.15)
        step3 = VGroup(step3_box, step3_text)

        # Position steps using grid row E
        self.place_at_grid(step1, "E1")
        self.place_at_grid(step2, "E3")
        self.place_at_grid(step3, "E5")

        # Create Arrows connecting the steps
        arrow1 = Arrow(start=self.grid["E1"], end=self.grid["E3"], buff=0.7, color=WHITE)
        arrow2 = Arrow(start=self.grid["E3"], end=self.grid["E5"], buff=0.7, color=WHITE)

        self.play(Create(step1))
        self.play(GrowArrow(arrow1))
        self.play(Create(step2))
        self.play(GrowArrow(arrow2))
        self.play(Create(step3))
        
        self.wait(3)