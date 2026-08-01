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
        # Setup the layout
        lecture_lines = [
            "Our formulas depend on zero factorial being one.",
            "Choosing three items from three requires zero factorial.",
            "If it were zero, the entire equation would implode."
        ]
        self.setup_layout("Validation: Keeping the Math Whole", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        # Using Text instead of MathTex to avoid 'latex' dependency
        self.lecture[0].set_color("#FFFFFF")
        formula = Text("nCr = n! / (r!(n-r)!)", color="#FFFFFF")
        self.place_in_area(formula, "A1", "B6", scale_factor=1.1)
        
        self.play(Write(formula))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Substitute n=3, r=3. Scenario A: 'If 0! = 1'.
        self.lecture[1].set_color("#00FF00")
        
        sub_formula = Text("3C3 = 3! / (3!(3-3)!) = 3! / (3! * 0!)", color="#FFFFFF")
        self.place_in_area(sub_formula, "C1", "D6", scale_factor=0.8)
        
        scenario_a = Text("If 0! = 1 -> 3C3 = 6 / (6 * 1) = 1", color="#00FF00")
        self.place_in_area(scenario_a, "E1", "E5", scale_factor=0.8)
        
        # Using a unicode checkmark instead of Tex
        checkmark = Text("✔", color="#00FF00")
        self.place_at_grid(checkmark, "E6", scale_factor=1.5)
        
        self.play(Write(sub_formula))
        self.wait(0.5)
        self.play(Write(scenario_a))
        self.play(FadeIn(checkmark))
        self.wait(1.5)

        # === Animation for Lecture Line 3 ===
        # Scenario B: 'If 0! = 0', denominator becomes zero.
        self.lecture[2].set_color("#FF0000")
        
        scenario_b = Text("If 0! = 0 -> 3! / (3! * 0) = 6 / 0...?", color="#FF0000")
        self.place_in_area(scenario_b, "F1", "F6", scale_factor=0.8)
        
        error_msg = Text("Division by Zero Error", color="#FF0000", weight=BOLD)
        self.place_in_area(error_msg, "B2", "B5", scale_factor=1.0)
        
        self.play(Write(scenario_b))
        self.wait(0.5)
        
        # Flash the error message
        self.play(Flash(error_msg, color="#FF0000", flash_radius=2, num_lines=12))
        self.add(error_msg)
        
        # Group components for destruction
        implode_group = VGroup(formula, sub_formula, scenario_a, checkmark, scenario_b, error_msg)
        
        # Shake effect - rapid small movements
        for _ in range(10):
            self.play(
                implode_group.animate.shift(np.array([np.random.uniform(-0.15, 0.15), np.random.uniform(-0.15, 0.15), 0])),
                run_time=0.04,
                rate_func=linear
            )

        # "Break" - pieces scatter randomly
        all_pieces = []
        for mob in implode_group:
            # All items are now Text or VGroup; Text is a subclass of VGroup
            if isinstance(mob, VGroup):
                all_pieces.extend(list(mob))
            else:
                all_pieces.append(mob)
        
        self.play(
            LaggedStart(
                *[
                    p.animate.shift(
                        np.array([np.random.uniform(-5, 5), np.random.uniform(-5, 5), 0])
                    ).rotate(np.random.uniform(-2*PI, 2*PI)).set_opacity(0)
                    for p in all_pieces
                ],
                lag_ratio=0.01
            ),
            run_time=1.5
        )
        
        self.wait(2)
