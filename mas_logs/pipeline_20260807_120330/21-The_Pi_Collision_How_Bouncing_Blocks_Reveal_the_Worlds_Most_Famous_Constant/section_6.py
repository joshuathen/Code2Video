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
        # Data from storyboard
        title_text = "Closing the Loop: Why Pi?"
        lecture_lines = [
            "As the large mass grows, the arcs shrink.",
            "Many tiny arcs approximate the circle's full circumference.",
            "The ratio of circumference to diameter is Pi.",
            "This explains why those specific digits appear.",
            "Dynamics and geometry reveal a fundamental constant."
        ]
        
        self.setup_layout(title_text, lecture_lines)
        
        # Colors
        COLOR_L1 = "#ADD8E6"  # Light Blue
        COLOR_L2 = "#90EE90"  # Light Green
        COLOR_L3 = "#F08080"  # Light Coral
        COLOR_GOLD = "#FFD700" # Gold
        COLOR_L5 = "#FFB6C1"  # Light Pink

        # === Animation for Lecture Line 1 ===
        # Highlight first lecture line
        self.play(self.lecture[0].animate.set_color(COLOR_L1))
        
        # Theta formula
        theta_formula = MathTex(r"\theta \approx \sqrt{\frac{m}{M}}", color=COLOR_L1)
        # Fix Issue 36: Relocate to avoid vertical crowding
        self.place_in_area(theta_formula, 'A1', 'B3', scale_factor=0.8)
        
        # Mass ratio tracker
        mass_ratio_tracker = ValueTracker(100)
        ratio_label = MathTex(r"M/m = ", color=COLOR_L1)
        ratio_value = DecimalNumber(mass_ratio_tracker.get_value(), num_decimal_places=0, color=COLOR_L1)
        ratio_group = VGroup(ratio_label, ratio_value).arrange(RIGHT)
        self.place_at_grid(ratio_group, "C2", scale_factor=0.9)
        
        # Persistent updater for decimal value
        ratio_value.add_updater(lambda d: d.set_value(mass_ratio_tracker.get_value()))
        
        self.play(Write(theta_formula), Write(ratio_group))
        # Mass ratio increases
        self.play(mass_ratio_tracker.animate.set_value(10000), run_time=2, rate_func=linear)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Update lecture highlight
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(COLOR_L2)
        )
        
        # Visualizing the circle of phases
        circle_outline = Circle(radius=1.2, color=WHITE, stroke_opacity=0.4)
        self.place_in_area(circle_outline, "C3", "E5", scale_factor=1.0)
        
        def create_arc_group(n_segments, color):
            grp = VGroup()
            angle_step = PI / n_segments
            # Performance safety: cap number of arc objects
            display_count = min(n_segments, 40)
            for i in range(display_count):
                arc = Arc(radius=1.2, start_angle=i*angle_step, angle=angle_step, color=color, stroke_width=4)
                arc.move_to(circle_outline.get_center())
                grp.add(arc)
            return grp

        arcs_initial = create_arc_group(8, COLOR_L2)
        arcs_final = create_arc_group(40, COLOR_L2)
        
        self.play(Create(circle_outline))
        self.play(Create(arcs_initial))
        self.wait(0.5)
        # Show higher density of arcs as theta decreases
        self.play(Transform(arcs_initial, arcs_final))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Update lecture highlight
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(COLOR_L3)
        )
        
        # The key formula N = floor(pi / theta)
        n_formula = MathTex(r"N = \lfloor \frac{\pi}{\theta} \rfloor", color=COLOR_L3)
        # Fix Issue 34: Relocate to avoid overlap with previous objects during transition
        self.place_in_area(n_formula, 'A4', 'B6', scale_factor=1.0)
        
        # Set Pi color to gold in formula
        n_formula.set_color_by_tex(r"\pi", COLOR_GOLD)
        
        self.play(
            FadeOut(theta_formula),
            FadeOut(ratio_group),
            FadeOut(circle_outline),
            FadeOut(arcs_initial),
            Write(n_formula)
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Update lecture highlight
        self.play(
            self.lecture[2].animate.set_color(WHITE),
            self.lecture[3].animate.set_color(COLOR_GOLD)
        )
        
        # Final digits reveal
        pi_digits = Text("3.14159265...", color=COLOR_GOLD, font_size=42)
        # Fix Issue 35: Relocate to bottom left visual area
        self.place_in_area(pi_digits, 'E1', 'F2', scale_factor=1.0)
        
        self.play(Write(pi_digits))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Update lecture highlight
        self.play(
            self.lecture[3].animate.set_color(WHITE),
            self.lecture[4].animate.set_color(COLOR_L5)
        )
        
        # Final summary relation
        summary = MathTex(r"M = 100^k \implies N \approx \pi \times 10^k", color=COLOR_L5)
        # Place summary in bottom right to clear pi_digits
        self.place_at_grid(summary, "F5", scale_factor=0.8)
        
        self.play(FadeIn(summary))
        self.wait(3)
