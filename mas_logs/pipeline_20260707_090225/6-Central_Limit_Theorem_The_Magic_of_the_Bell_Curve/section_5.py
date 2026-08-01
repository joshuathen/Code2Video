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
        # Setup layout with title and lecture script
        lecture_lines = [
            'The mean of averages equals the original population mean.', 
            'Larger samples create a much narrower and taller curve.', 
            'This increased precision reduces our overall margin of error.', 
            'Typically, a sample size of thirty is the magic number.', 
            'Mathematics provides certainty even when starting with chaos.'
        ]
        self.setup_layout("The Mathematical Pillars", lecture_lines)

        # --- Initialization ---
        # ValueTracker for the sample size n
        n_tracker = ValueTracker(4)
        mu_val = 0
        sigma_pop = 1.0

        # Axes: Located in the center-right area (B1 to F6)
        axes = Axes(
            x_range=[-3.5, 3.5, 1],
            y_range=[0, 5, 1],
            axis_config={"include_tip": False},
            x_length=4.5,
            y_length=4
        )
        self.place_in_area(axes, "B1", "F6")

        # Mu Center Highlight (#FFD700)
        mu_line = Line(
            axes.c2p(mu_val, 0), axes.c2p(mu_val, 4.5), 
            color="#FFD700", stroke_width=3
        )
        # Using Text for mu label to avoid potential LaTeX dependency issues
        mu_label = Text("μ", color="#FFD700", font_size=32)
        mu_label.next_to(mu_line, UP, buff=0.1)

        # Standard Error Formula (#FFFFFF)
        # ISSUE 43/51 Fix: Move formula to area A3-A4 to avoid overcrowding near labels
        formula = Text("SE = σ / √n", color=WHITE, font_size=26)
        self.place_in_area(formula, 'A3', 'A4', scale_factor=0.8)
        
        # Sample size indicator for the animation
        # ISSUE 44/51 Fix: Move n_display to grid A5 for better visual alignment in the top row
        n_val_label = Text("n = ", font_size=24, color=WHITE)
        n_val_num = DecimalNumber(n_tracker.get_value(), num_decimal_places=0, color=YELLOW, font_size=28, mob_class=Text)
        n_val_num.add_updater(lambda d: d.set_value(n_tracker.get_value()))
        n_display = VGroup(n_val_label, n_val_num).arrange(RIGHT, buff=0.2)
        self.place_at_grid(n_display, "A5", scale_factor=0.9)

        # Magic Number Asset & Text (#00FF00)
        # ISSUE 28/51: Integrated [Asset: /mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/based.svg]
        magic_number_txt = Text("Magic Number: n ≥ 30", color="#00FF00", font_size=26)
        based_icon = SVGMobject("/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/based.svg")
        based_icon.set_color("#00FF00").scale(0.3)
        magic_number_group = VGroup(based_icon, magic_number_txt).arrange(RIGHT, buff=0.2)
        # ISSUE 42/51 Fix: Relocated to area A1-A2 to prevent overlap with x-axis
        self.place_in_area(magic_number_group, 'A1', 'A2', scale_factor=0.7)

        # Dynamic Bell Curve function (using dynamic n via ValueTracker)
        # ISSUE 51: Ensure dynamic narrowness reflects n increase
        def get_bell_curve():
            n = n_tracker.get_value()
            se = sigma_pop / np.sqrt(n)
            # Normal distribution probability density function plot
            return axes.plot(
                lambda x: (1 / (se * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu_val) / se) ** 2),
                color=BLUE,
                x_range=[-3.5, 3.5]
            )

        # Persistent mobject with redrawer
        curve = always_redraw(get_bell_curve)

        # === Animation for Lecture Line 1 ===
        # 'The mean of averages equals the original population mean.'
        self.lecture[0].set_color(YELLOW)
        self.play(Create(axes))
        self.play(Create(mu_line), Write(mu_label))
        self.play(Create(curve), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # 'Larger samples create a much narrower and taller curve.'
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        self.play(Write(formula), Write(n_display))
        # Animate n from 4 to 30 to show the 'Magic Number' transition
        self.play(n_tracker.animate.set_value(30), run_time=3, rate_func=linear)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # 'This increased precision reduces our overall margin of error.'
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        # Animate n further to 100 to show extreme precision and tall/narrow curve
        self.play(n_tracker.animate.set_value(100), run_time=2, rate_func=linear)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # 'Typically, a sample size of thirty is the magic number.'
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        # Reveal the Magic Number text and the based icon asset
        self.play(FadeIn(based_icon), Write(magic_number_txt))
        self.play(Indicate(magic_number_group, color="#00FF00", scale_factor=1.1))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # 'Mathematics provides certainty even when starting with chaos.'
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        # Highlight mu as the unchanging center of the distribution
        self.play(Indicate(mu_line, color="#FFD700"), Flash(mu_line.get_top(), color="#FFD700"))
        self.wait(2)
        
        # Cleanup
        self.lecture[4].set_color(WHITE)
        self.wait(1)
