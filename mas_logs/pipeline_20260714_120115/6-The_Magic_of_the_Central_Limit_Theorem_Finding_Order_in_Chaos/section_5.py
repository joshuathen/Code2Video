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

class Section5Scene(Scene):
    def construct(self):
        # 1. Title and Header
        title = Text("The Magic of the Central Limit Theorem", font_size=36).to_edge(UP)
        subtitle = Text("Finding Order in Chaos", font_size=28, color=BLUE).next_to(title, DOWN)
        
        self.play(Write(title))
        self.play(FadeIn(subtitle))
        self.wait(1)

        # 2. Setup Axes
        axes = Axes(
            x_range=[-4, 4, 1],
            y_range=[0, 0.5, 0.1],
            x_length=10,
            y_length=4.5,
            axis_config={"include_tip": False},
        ).shift(DOWN * 0.5)

        # FIX: Pass Text mobjects instead of strings to avoid LaTeX dependency (FileNotFoundError: 'latex')
        x_label = axes.get_x_axis_label(Text("Value", font_size=24))
        y_label = axes.get_y_axis_label(Text("Frequency", font_size=24))

        self.play(Create(axes), Write(x_label), Write(y_label))

        # 3. Representing "Chaos" (Initial Random Distribution)
        num_bars = 24
        bar_width = 8.0 / num_bars
        np.random.seed(42) # For consistency
        chaos_heights = np.random.uniform(0.05, 0.4, num_bars)
        
        chaos_bars = VGroup()
        for i, h in enumerate(chaos_heights):
            x_pos = -4 + i * bar_width + bar_width/2
            # Calculate height in scene units
            h_unit = axes.coords_to_point(0, h)[1] - axes.coords_to_point(0, 0)[1]
            bar = Rectangle(
                width=bar_width * 0.9,
                height=h_unit,
                fill_opacity=0.6,
                fill_color=GREY,
                stroke_width=0.5,
                stroke_color=WHITE
            )
            bar.move_to(axes.coords_to_point(x_pos, 0), aligned_edge=DOWN)
            chaos_bars.add(bar)

        self.play(Create(chaos_bars))
        self.wait(1)

        # 4. Representing "Order" (The Normal Distribution)
        mu = 0
        sigma = 1
        
        def normal_pdf(x):
            return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma)**2)

        order_bars = VGroup()
        for i in range(num_bars):
            x_pos = -4 + i * bar_width + bar_width/2
            h = normal_pdf(x_pos)
            h_unit = axes.coords_to_point(0, h)[1] - axes.coords_to_point(0, 0)[1]
            bar = Rectangle(
                width=bar_width * 0.9,
                height=h_unit,
                fill_opacity=0.8,
                fill_color=BLUE,
                stroke_width=0.5,
                stroke_color=WHITE
            )
            bar.move_to(axes.coords_to_point(x_pos, 0), aligned_edge=DOWN)
            order_bars.add(bar)

        normal_curve = axes.plot(
            lambda x: normal_pdf(x),
            color=YELLOW,
            stroke_width=4
        )

        # 5. Transition: Chaos to Order
        self.play(
            Transform(chaos_bars, order_bars),
            subtitle.animate.set_color(YELLOW),
            run_time=2.5
        )
        self.play(Create(normal_curve))
        self.wait(0.5)

        # 6. CLT Formula
        # FIX: Replaced MathTex with Text to remove dependency on external 'latex' compiler
        clt_formula = Text(
            "X_n → N(μ, σ²/n)",
            font_size=30,
            color=YELLOW
        ).to_corner(UR, buff=1.0)

        background_rect = SurroundingRectangle(clt_formula, color=WHITE, fill_opacity=0.2, buff=0.2)
        formula_group = VGroup(background_rect, clt_formula)

        self.play(
            FadeIn(formula_group),
            title.animate.scale(0.8).to_edge(UL)
        )
        self.wait(1)

        # 7. Final Conclusion
        conclusion = Text(
            "Universal patterns emerge from independent random variables.",
            font_size=22,
            line_spacing=1.5
        ).to_edge(DOWN, buff=0.3)
        
        self.play(FadeIn(conclusion))
        self.wait(3)
