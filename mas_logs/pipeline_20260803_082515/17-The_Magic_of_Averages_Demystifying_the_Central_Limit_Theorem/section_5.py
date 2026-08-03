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
        # Setup
        title = "Mathematical Nuance: Mean and Standard Error"
        lines = [
            "The sample mean's center matches the original population mean.",
            "As n grows, the bell curve becomes much narrower.",
            "This narrowing is called the standard error.",
            "Formula: standard deviation divided by square root of n.",
            "Larger samples provide much more precise estimates."
        ]
        self.setup_layout(title, lines)

        # Colors
        COLOR_POP = "#FF8C00" # Orange
        COLOR_SAMPLE = "#00FFFF" # Cyan
        COLOR_FORMULA = "#FFFF00" # Yellow
        COLOR_TEXT = "#FFFFFF" # White

        # Normal Distribution Function
        def normal_pdf(x, mu, sigma):
            return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma)**2)

        # Base parameters
        mu_val = 0
        sigma_pop = 1.0
        n_tracker = ValueTracker(1)

        # Grid-based coordinate references
        # x_center is halfway between Col 3 and 4
        x_center = (self.grid["F3"][0] + self.grid["F4"][0]) / 2
        # y_base is on Row F
        y_base = self.grid["F1"][1]

        # X-axis
        x_axis = Line(
            start=np.array([self.grid["F1"][0] - 0.3, y_base, 0]),
            end=np.array([self.grid["F6"][0] + 0.3, y_base, 0]),
            color=WHITE,
            stroke_width=2
        )

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(COLOR_POP)
        
        # Population Curve
        pop_curve = FunctionGraph(
            lambda x: normal_pdf(x, 0, sigma_pop),
            x_range=[-3, 3, 0.05],
            color=COLOR_POP
        )
        pop_curve.move_to([x_center, y_base, 0], aligned_edge=DOWN)

        # Sample Mean Curve (initially matches pop)
        sample_curve = pop_curve.copy().set_color(COLOR_SAMPLE)

        # Vertical dashed line at peaks
        peak_x = x_center
        peak_y_top = self.grid["A3"][1] + 0.5
        peak_y_bottom = y_base
        
        center_line = DashedLine(
            start=[peak_x, peak_y_bottom, 0],
            end=[peak_x, peak_y_top, 0],
            color=COLOR_TEXT
        )
        mu_label = MathTex(r"\mu", color=COLOR_TEXT, font_size=32)
        mu_label.next_to(center_line, UP, buff=0.1)

        self.add(x_axis)
        self.play(Create(pop_curve), run_time=1)
        self.play(Create(sample_curve), run_time=1)
        self.play(Create(center_line), Write(mu_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(COLOR_SAMPLE)
        
        # n label - Addressing Issue 34 (Move to A4-A6)
        n_label_text = Text("n = ", font_size=24, color=COLOR_TEXT)
        n_value = DecimalNumber(1, num_decimal_places=0, color=COLOR_TEXT, font_size=24)
        n_group = VGroup(n_label_text, n_value).arrange(RIGHT, buff=0.1)
        self.place_in_area(n_group, "A4", "A6", scale_factor=0.8)

        n_value.add_updater(lambda d: d.set_value(n_tracker.get_value()))
        
        # Updater for cyan curve
        def update_sample_curve(mob):
            n_val = n_tracker.get_value()
            if n_val < 1: n_val = 1
            sigma_sample = sigma_pop / np.sqrt(n_val)
            # Higher resolution for narrow curves
            new_mob = FunctionGraph(
                lambda x: normal_pdf(x, 0, sigma_sample),
                x_range=[-3, 3, 0.01],
                color=COLOR_SAMPLE
            )
            new_mob.move_to([x_center, y_base, 0], aligned_edge=DOWN)
            mob.become(new_mob)

        sample_curve.add_updater(update_sample_curve)
        
        self.add(n_group)
        self.play(n_tracker.animate.set_value(16), run_time=2, rate_func=linear)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(COLOR_SAMPLE)
        
        # Indicator for width (Standard Error)
        # Using persistent arrow with updater for performance
        se_arrow = DoubleArrow(
            start=[x_center, y_base + 0.3, 0],
            end=[x_center + 1.0, y_base + 0.3, 0],
            buff=0,
            color=COLOR_TEXT,
            stroke_width=2,
            max_tip_length_to_length_ratio=0.2
        )
        
        def update_se_arrow(mob):
            n_val = n_tracker.get_value()
            sigma_se = sigma_pop / np.sqrt(n_val)
            y_pos = y_base + 0.3
            mob.put_start_and_end_on([x_center, y_pos, 0], [x_center + sigma_se, y_pos, 0])
            
        se_arrow.add_updater(update_se_arrow)
        
        se_text = Text("Standard Error", font_size=18, color=COLOR_TEXT)
        se_text.add_updater(lambda t: t.next_to(se_arrow, UP, buff=0.05))
        
        self.play(Create(se_arrow), Write(se_text))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(COLOR_FORMULA)
        
        # Formula - Addressing Issue 33 (Move to A1-A3)
        formula = MathTex(
            r"SE = \frac{\sigma}{\sqrt{n}}",
            color=COLOR_FORMULA,
            font_size=32
        )
        self.place_in_area(formula, "A1", "A3", scale_factor=0.8)
        
        self.play(Write(formula))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(COLOR_TEXT)
        
        # Final n increase to show extreme narrowing
        self.play(
            n_tracker.animate.set_value(81),
            run_time=3,
            rate_func=smooth
        )
        self.wait(2)
