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
        # Data from shared state
        title = "Defining the Zeta Function"
        lines = [
            "The zeta function sums one over n to power s.",
            "At s equals one, the harmonic series explodes infinitely.",
            "As s increases, the sum settles into finite values.",
            "For s equals two, we solve the famous Basel Problem.",
            "The function converges for all values greater than one."
        ]
        self.setup_layout(title, lines)

        # Asset Paths
        tower_path = "/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/tower.svg"
        scale_path = "/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/scale.svg"

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(WHITE))
        # Replaced MathTex with Text to avoid LaTeX dependency
        formula = Text("ζ(s) = ∑ 1/n^s", font_size=32, color=WHITE)
        # Fixed: Issue 36 - repositioned formula
        self.place_in_area(formula, 'A1', 'B3', scale_factor=1.0)
        self.play(Write(formula))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        l2_color = BLUE_A
        self.play(self.lecture[1].animate.set_color(l2_color))
        
        s_tracker = ValueTracker(2.0)
        
        # Slider UI - Added label_constructor=Text to avoid LaTeX dependency
        slider_line = NumberLine(x_range=[1, 4, 1], length=2.4, include_numbers=True, font_size=16, label_constructor=Text)
        self.place_at_grid(slider_line, "C3", scale_factor=0.9)
        
        # UI labels
        s_label = Text("s = ", font_size=24)
        # Added mob_class=Text to DecimalNumber to avoid LaTeX dependency
        s_val_disp = DecimalNumber(s_tracker.get_value(), num_decimal_places=1, font_size=24, mob_class=Text)
        s_val_disp.add_updater(lambda d: d.set_value(s_tracker.get_value()))
        slider_ctrl = VGroup(s_label, s_val_disp).arrange(RIGHT, buff=0.1).next_to(slider_line, UP, buff=0.1)
        
        slider_dot = Dot(color=YELLOW, radius=0.08)
        slider_dot.add_updater(lambda d: d.move_to(slider_line.n2p(s_tracker.get_value())))
        
        # Tower Visual (Asset)
        tower_icon = SVGMobject(tower_path).set_color(BLUE_E).set_opacity(0.2)
        # Fixed: Issue 35 - repositioned tower_icon
        self.place_in_area(tower_icon, 'D1', 'F2', scale_factor=1.1)
        
        # Dynamic Blocks
        blocks = VGroup(*[Rectangle(width=0.4, height=0.1, fill_opacity=0.8, color=l2_color, stroke_width=1) for _ in range(5)])
        
        def update_blocks(m):
            s_val = s_tracker.get_value()
            base_point = tower_icon.get_bottom()
            center_x = base_point[0]
            curr_y = base_point[1]
            for i, b in enumerate(m):
                n = i + 1
                h = 1.3 / (n**s_val)
                b.stretch_to_fit_height(max(h, 0.02))
                b.move_to([center_x, curr_y + b.get_height()/2, 0])
                curr_y += b.get_height()
        
        blocks.add_updater(update_blocks)
        
        self.play(FadeIn(slider_line), FadeIn(slider_ctrl), FadeIn(slider_dot))
        self.play(FadeIn(tower_icon), FadeIn(blocks))
        
        # Animate to Harmonic Series (s=1)
        self.play(s_tracker.animate.set_value(1.0), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        l3_color = TEAL
        self.play(self.lecture[2].animate.set_color(l3_color))
        # Converging to higher s
        self.play(s_tracker.animate.set_value(3.0), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        l4_color = "#FFD700" # Gold
        self.play(self.lecture[3].animate.set_color(l4_color))
        
        # Return to s=2 for Basel visual as implied by storyboard
        self.play(s_tracker.animate.set_value(2.0), run_time=0.8)
        self.play(FadeOut(slider_line, slider_ctrl, slider_dot, tower_icon, blocks), run_time=0.5)
        
        # Balance Scale (Asset)
        scale_icon = SVGMobject(scale_path).set_color(GRAY)
        # Fixed: Issue 35 - repositioned scale_icon
        self.place_in_area(scale_icon, 'D4', 'F6', scale_factor=1.1)
        
        # Basel Squares
        basel_sqs = VGroup(*[
            Square(side_length=0.6/(n), fill_opacity=0.7, color=TEAL)
            for n in range(1, 5)
        ]).arrange(RIGHT, buff=0.05, aligned_edge=DOWN)
        basel_sqs.move_to(scale_icon.get_center() + LEFT * 0.8 + UP * 0.4)
        
        # Specific solution
        basel_result = Text("ζ(2) = π²/6", font_size=32, color=l4_color)
        # Fixed: Issue 37 - repositioned basel_result
        self.place_at_grid(basel_result, 'C5', scale_factor=1.0)
        
        self.play(FadeIn(scale_icon), FadeIn(basel_sqs))
        self.play(Write(basel_result))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        l5_color = "#90EE90" # Light Green
        self.play(self.lecture[4].animate.set_color(l5_color))
        
        conv_text = Text("Convergence: s > 1", font_size=22, color=l5_color)
        conv_box = SurroundingRectangle(conv_text, color=l5_color, buff=0.2)
        conv_label = VGroup(conv_box, conv_text)
        # Fixed: Issue 36 - repositioned conv_label
        self.place_at_grid(conv_label, 'A5', scale_factor=0.9)
        
        self.play(Create(conv_box), Write(conv_text))
        self.wait(2)
