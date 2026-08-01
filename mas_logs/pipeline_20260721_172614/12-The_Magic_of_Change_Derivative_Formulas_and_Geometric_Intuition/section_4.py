from manim import *

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
        # Title and Lecture Lines
        title_text = "The Power Rule: Symbolic Shortcut"
        lecture_lines = [
            "Calculating limits every time is very slow.",
            "The Power Rule gives us a faster shortcut.",
            "Take the exponent and move it to the front.",
            "Then, subtract one from the original exponent.",
            "For x squared, the derivative is two x.",
        ]
        
        self.setup_layout(title_text, lecture_lines)
        
        # Colors
        BLUE_SHORTCUT = "#58C4DD"
        ORANGE_EXP = "#FF8000"
        HIGHLIGHT_COLOR = YELLOW # Standard highlight for white text
        
        # === Animation for Lecture Line 1 ===
        # Display 'x to the power of n' in white (#FFFFFF) in the center of the screen.
        self.play(self.lecture[0].animate.set_color(HIGHLIGHT_COLOR))
        
        base_x = MathTex("x", color=WHITE)
        # Fix Issue 30: Moved from B3 to B2 and adjusted scale
        self.place_at_grid(base_x, "B2", scale_factor=1.2)
        
        exp_n = MathTex("n", color=WHITE).scale(1.2 * 0.7)
        exp_n.next_to(base_x.get_corner(UR), UR, buff=0.05)
        
        power_expr = VGroup(base_x, exp_n)
        
        self.play(Write(power_expr))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Show a 'shortcut' arrow (#58C4DD) pointing to the side.
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(BLUE_SHORTCUT)
        )
        
        # Positioned arrow to the right of B2 (starts at B3 area) to avoid overlap
        arrow = Arrow(
            start=self.grid["B3"] + LEFT * 0.2,
            end=self.grid["B5"] + RIGHT * 0.2,
            color=BLUE_SHORTCUT,
            buff=0.1,
            stroke_width=6
        )
        shortcut_lbl = Text("Shortcut", font_size=18, color=BLUE_SHORTCUT)
        shortcut_lbl.next_to(arrow, UP, buff=0.1)
        
        self.play(GrowArrow(arrow), FadeIn(shortcut_lbl))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Animate the 'n' moving from the exponent to the front of 'x' and changing color to bright orange (#FF8000).
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(ORANGE_EXP)
        )
        
        target_front_pos = base_x.get_left() + LEFT * 0.35
        
        self.play(
            exp_n.animate.move_to(target_front_pos).set_color(ORANGE_EXP).scale(1/0.7),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Change the remaining exponent to 'n minus 1' in orange (#FF8000).
        self.play(
            self.lecture[2].animate.set_color(WHITE),
            self.lecture[3].animate.set_color(ORANGE_EXP)
        )
        
        new_exp = MathTex("n-1", color=ORANGE_EXP).scale(1.2 * 0.7)
        new_exp.next_to(base_x.get_corner(UR), UR, buff=0.05)
        
        self.play(FadeIn(new_exp, shift=DOWN * 0.2))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Display 'd/dx(x^2) = 2x' as a specific example below the rule.
        self.play(
            self.lecture[3].animate.set_color(WHITE),
            self.lecture[4].animate.set_color(HIGHLIGHT_COLOR)
        )
        
        example = MathTex(r"\frac{d}{dx}(x^2) = 2x", color=WHITE)
        # Fix Issue 31: Moved from D3 to E3 and adjusted scale
        self.place_at_grid(example, "E3", scale_factor=1.1)
        
        self.play(Write(example))
        self.wait(2)
        
        # Cleanup
        self.play(self.lecture[4].animate.set_color(WHITE))
        self.wait(2)
