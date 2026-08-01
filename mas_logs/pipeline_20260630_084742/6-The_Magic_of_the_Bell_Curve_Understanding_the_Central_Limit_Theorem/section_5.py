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
        # Setup the title and lecture lines
        lecture_lines = [
            "First, the shape always becomes a bell curve.",
            "Second, the center stays at the population mean.",
            "Third, larger samples make the estimate much more precise."
        ]
        self.setup_layout("The Three Pillars of the Theorem", lecture_lines)

        # Define Colors
        COLOR_NORMAL = WHITE
        COLOR_GOLD = "#FFD54F"
        COLOR_WIDE = "#EF5350"  # Red
        COLOR_NARROW = "#66BB6A" # Green

        # Create Axes
        axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[0, 4, 1],
            axis_config={"include_tip": False, "stroke_width": 2},
            x_length=5,
            y_length=4,
        ).set_color(GRAY_C)
        self.place_in_area(axes, "A1", "F6")

        # Gaussian Function
        def gaussian(x, mu, sigma, amplitude):
            return amplitude * np.exp(-0.5 * ((x - mu) / sigma)**2)

        # === Animation for Lecture Line 1 ===
        # Integrate Asset: bell.svg
        bell_icon = SVGMobject("/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/bell.svg").set_color(WHITE)
        self.place_at_grid(bell_icon, "A1", scale_factor=0.4)
        
        label_normal = Text("Normal", color=WHITE)
        self.place_at_grid(label_normal, "A2", scale_factor=0.6)
        
        # Base normal curve for context
        curve_white = axes.plot(
            lambda x: gaussian(x, 0, 0.8, 2.5),
            color=WHITE,
            stroke_width=2
        )

        self.play(self.lecture[0].animate.set_color(WHITE))
        self.play(
            Create(axes),
            Create(curve_white),
            Write(label_normal),
            FadeIn(bell_icon),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Vertical gold line for mean at center
        mean_line = axes.get_vertical_line(
            axes.c2p(0, 3.5), color=COLOR_GOLD, line_func=DashedLine
        )
        # Mu label using Text for consistency
        mu_label = Text("μ", color=COLOR_GOLD)
        self.place_at_grid(mu_label, "E4", scale_factor=0.8)

        self.play(self.lecture[1].animate.set_color(COLOR_GOLD))
        self.play(
            Create(mean_line),
            Write(mu_label),
            run_time=1
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Red wide curve (n=5)
        curve_wide = axes.plot(
            lambda x: gaussian(x, 0, 1.2, 1.5),
            color=COLOR_WIDE,
            stroke_width=4
        )
        label_n5 = Text("n=5", color=COLOR_WIDE)
        self.place_at_grid(label_n5, "C5", scale_factor=0.8) # Issue 45 Fix

        # Green narrow curve (n=50)
        curve_narrow = axes.plot(
            lambda x: gaussian(x, 0, 0.4, 3.5),
            color=COLOR_NARROW,
            stroke_width=4
        )
        label_n50 = Text("n=50", color=COLOR_NARROW)
        self.place_at_grid(label_n50, "B2", scale_factor=0.8) # Issue 46 Fix

        # Precision Lens Graphic
        lens_circle = Circle(radius=0.3, color=WHITE, stroke_width=2)
        lens_handle = Line(start=ORIGIN, end=RIGHT*0.2, color=WHITE, stroke_width=2).rotate(-PI/4)
        lens_handle.next_to(lens_circle, DR, buff=-0.05)
        precision_lens = VGroup(lens_circle, lens_handle)
        
        # Move lens to high-precision peak area
        self.place_in_area(precision_lens, "B3", "B4", scale_factor=1.0)
        
        lens_text = Text("High Precision", color=COLOR_NARROW)
        self.place_in_area(lens_text, "A3", "A4", scale_factor=0.6) # Issue 44 Fix

        self.play(self.lecture[2].animate.set_color(COLOR_NARROW))
        self.play(
            Create(curve_wide),
            Write(label_n5),
            run_time=1.2
        )
        self.play(
            Create(curve_narrow),
            Write(label_n50),
            run_time=1.2
        )
        self.play(
            FadeIn(precision_lens),
            Write(lens_text),
            run_time=1
        )
        self.play(
            Indicate(precision_lens, scale_factor=1.1, color=COLOR_NARROW),
            run_time=1
        )
        self.wait(2)
