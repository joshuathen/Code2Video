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

class Section4Scene(TeachingScene):
    def construct(self):
        # Data from storyboard
        lecture_lines = [
            "Watch as the collection of averages begins to change.",
            "The chaotic forest data transforms into a symmetrical shape.",
            "A perfect bell curve emerges from the underlying chaos.",
            "This magic happens regardless of the original distribution's shape.",
            "With enough samples, the Central Limit Theorem always wins."
        ]
        
        self.setup_layout("The Core Revelation: The Bell Curve Emerges", lecture_lines)

        # Colors
        COLOR_FOREST = "#888888"   # Grey for "chaos"
        COLOR_MEAN = "#3498DB"     # Blue for sample means
        COLOR_BELL = "#00FF00"     # Green for the bell curve
        COLOR_CLT = "#FFFFFF"      # White for CLT text
        COLOR_CONDITION = "#F1C40F" # Yellow for n>=30
        COLOR_SHAPES = "#FF69B4"   # Pink for alternate shapes

        # Assets
        FOREST_SVG_PATH = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/forest.svg"

        # === Animation for Lecture Line 1 ===
        # Line 1: Watch as the collection of averages begins to change.
        self.lecture[0].set_color(WHITE)

        # Forest Distribution (top left)
        # Using Critic Fix: A1-D3, scale 0.9
        forest_icon = SVGMobject(FOREST_SVG_PATH, height=0.6).set_color(COLOR_FOREST)
        forest_axes = Axes(
            x_range=[0, 10, 2], y_range=[0, 1, 0.2],
            x_length=2.2, y_length=1.5,
            axis_config={"include_tip": False, "font_size": 12}
        ).set_color(GRAY_C)
        forest_label = Text("Forest Dist.", font_size=16, color=GRAY_A)
        forest_plot = forest_axes.plot(
            lambda x: 0.4 * np.exp(-0.5 * (x - 2)**2) + 0.6 * np.exp(-0.5 * (x - 7)**2),
            color=COLOR_FOREST
        )
        forest_group = VGroup(forest_icon, forest_axes, forest_plot, forest_label).arrange(DOWN, buff=0.1)
        self.place_in_area(forest_group, 'A1', 'D3', scale_factor=0.9)

        # Sample Mean Distribution (top right)
        # Using Critic Fix: A4-D6, scale 0.9
        mean_axes = Axes(
            x_range=[0, 10, 2], y_range=[0, 1, 0.2],
            x_length=2.2, y_length=1.5,
            axis_config={"include_tip": False, "font_size": 12}
        ).set_color(GRAY_C)
        mean_label = Text("Sample Mean Dist.", font_size=16, color=COLOR_MEAN)
        mean_group = VGroup(mean_axes, mean_label).arrange(DOWN, buff=0.1)
        self.place_in_area(mean_group, 'A4', 'D6', scale_factor=0.9)

        self.play(FadeIn(forest_group), Create(mean_axes), Write(mean_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Line 2: The chaotic forest data transforms into a symmetrical shape.
        self.lecture[1].set_color(COLOR_MEAN)

        # Generate 500 dots stacked in a bell-like histogram
        num_dots = 500
        num_bins = 18
        bin_edges = np.linspace(2, 8, num_bins + 1)
        mu, sigma = 5, 1.1
        
        # Seed for consistency
        np.random.seed(42)
        samples = np.random.normal(mu, sigma, num_dots)
        samples = np.clip(samples, 2.1, 7.9)
        
        bins = [[] for _ in range(num_bins)]
        for s in samples:
            for i in range(num_bins):
                if bin_edges[i] <= s < bin_edges[i+1]:
                    bins[i].append(s)
                    break
        
        dots = VGroup()
        dot_radius = 0.025
        for i, bin_samples in enumerate(bins):
            x_center = (bin_edges[i] + bin_edges[i+1]) / 2
            for j, _ in enumerate(bin_samples):
                # Stack dots vertically in the bin
                # y_unit is small, we scale it
                pos = mean_axes.c2p(x_center, j * 0.015) 
                dot = Dot(point=pos, radius=dot_radius, color=COLOR_MEAN, fill_opacity=0.7)
                dots.add(dot)

        # Rapidly fill in batches
        batch_size = 50
        for i in range(0, num_dots, batch_size):
            batch = dots[i:i+batch_size]
            self.play(FadeIn(batch, lag_ratio=0.01), run_time=0.25)
        
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Line 3: A perfect bell curve emerges from the underlying chaos.
        self.lecture[2].set_color(COLOR_BELL)

        bell_curve = mean_axes.plot(
            lambda x: 0.8 * np.exp(-0.5 * ((x - 5) / 1.1)**2),
            color=COLOR_BELL,
            stroke_width=4
        )
        
        self.play(Create(bell_curve))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Line 4: This magic happens regardless of the original distribution's shape.
        self.lecture[3].set_color(COLOR_SHAPES)

        # Show icons of various shapes transforming to bell curves
        # Placing in the empty area E1-F4 to avoid collisions
        uniform_rect = Rectangle(height=0.4, width=0.6, color=COLOR_SHAPES, fill_opacity=0.4)
        skewed_shape = Polygon(
            [0,0,0], [0.1, 0.4, 0], [0.3, 0.2, 0], [0.6, 0.1, 0], [0.6, 0, 0],
            color=COLOR_SHAPES, fill_opacity=0.4
        )
        
        icon1 = uniform_rect.copy()
        icon2 = skewed_shape.copy()
        
        self.place_at_grid(icon1, 'E1', scale_factor=1.0)
        self.place_at_grid(icon2, 'E3', scale_factor=1.0)
        
        small_bell_template = mean_axes.plot(
            lambda x: 0.5 * np.exp(-0.5 * ((x - 5) / 0.5)**2),
            color=COLOR_BELL,
            stroke_width=2
        ).scale(0.15)
        
        self.play(FadeIn(icon1), FadeIn(icon2))
        self.wait(0.5)
        
        target1 = small_bell_template.copy().move_to(icon1)
        target2 = small_bell_template.copy().move_to(icon2)
        
        self.play(
            Transform(icon1, target1),
            Transform(icon2, target2)
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Line 5: With enough samples, the Central Limit Theorem always wins.
        self.lecture[4].set_color(COLOR_CONDITION)

        clt_text = Text("Central Limit Theorem", font_size=24, color=COLOR_CLT, weight=BOLD)
        # Using a safe area that doesn't overlap the charts or the icons
        # Charts are in rows A-D. Icons are at E1, E3. Condition at E5-F6.
        # Put CLT text in Row F, left side.
        self.place_in_area(clt_text, 'F1', 'F3', scale_factor=0.9)
        
        condition = MathTex("n \\geq 30", font_size=36, color=COLOR_CONDITION)
        self.place_in_area(condition, 'E5', 'F6', scale_factor=0.8) # Critic Fix (Issue 30)

        self.play(Write(clt_text), Write(condition))
        
        # Final pulse animation
        self.play(
            bell_curve.animate.set_stroke(width=8),
            clt_text.animate.scale(1.15),
            rate_func=there_and_back,
            run_time=1.5
        )
        self.play(
            bell_curve.animate.set_stroke(width=4),
            clt_text.animate.scale(1/1.15),
            rate_func=there_and_back,
            run_time=1.5
        )
        
        self.wait(2)
