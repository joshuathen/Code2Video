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
        lecture_lines_text = [
            'Visualize a filter window sliding across a stationary signal.',
            'The overlapping area determines the output at each index.',
            'This sliding process creates a modified version of data.'
        ]
        self.setup_layout("Graphical Interpretation: The Sliding Window", lecture_lines_text)

        # Colors
        SIGNAL_COLOR = "#58C4DD"
        FILTER_COLOR = "#F9D71C"
        OVERLAP_COLOR = "#FFFFFF" 
        RESULT_COLOR = "#FF69B4"

        # Signal Data (Stationary f[k])
        signal_vals = [0, 0.4, 0.8, 1.5, 2.0, 1.4, 0.8, 0.5, 0.3, 0.1, 0]
        
        # Upper Axes (Top Area: A1-C6)
        ax_top = Axes(
            x_range=[-2, 12, 2],
            y_range=[0, 3, 1],
            x_length=5,
            y_length=2.5,
            axis_config={"include_tip": False, "color": GREY},
            tips=False
        )
        self.place_in_area(ax_top, 'A1', 'C6', scale_factor=1.0)
        
        bars_f = VGroup(*[
            Rectangle(
                width=0.3, height=val, 
                fill_opacity=0.6, fill_color=SIGNAL_COLOR, stroke_width=1
            ).move_to(ax_top.c2p(i, val/2))
            for i, val in enumerate(signal_vals)
        ])
        
        # Use Text instead of MathTex to avoid FileNotFoundError: 'latex'
        f_label = Text("f[k]", color=SIGNAL_COLOR)
        self.place_at_grid(f_label, 'A1', scale_factor=0.7)

        # Lower Axes (Bottom Area: D1-F6)
        ax_bottom = Axes(
            x_range=[-2, 12, 2],
            y_range=[0, 3, 1],
            x_length=5,
            y_length=2.5,
            axis_config={"include_tip": False, "color": GREY},
            tips=False
        )
        self.place_in_area(ax_bottom, 'D1', 'F6', scale_factor=1.0)
        
        # Use Text instead of MathTex to avoid FileNotFoundError: 'latex'
        y_label = Text("y[n]", color=RESULT_COLOR)
        self.place_at_grid(y_label, 'D1', scale_factor=0.7)

        # Tracker for sliding n
        n_tracker = ValueTracker(-2)

        # Assets
        window_icon = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/window.svg")
        window_icon.set_color(WHITE).set_stroke(width=2)
        
        # Filter Kernel g[n-k] representation (a semi-transparent yellow box inside the window)
        kernel_rect = Rectangle(width=0.8, height=2.0, fill_opacity=0.3, fill_color=FILTER_COLOR, stroke_color=FILTER_COLOR)
        window_group = VGroup(window_icon, kernel_rect).scale(0.35)
        
        def window_updater(m):
            m.move_to(ax_top.c2p(n_tracker.get_value(), 1.0))
        window_group.add_updater(window_updater)

        # Convolution calculation helper
        def get_conv_val(n):
            total = 0
            # Averaging window size 3
            for k in range(n-1, n+2):
                if 0 <= k < len(signal_vals):
                    total += signal_vals[k]
            return total / 3.0

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(FILTER_COLOR)
        self.add(ax_top, bars_f, f_label, window_group)
        self.play(n_tracker.animate.set_value(10), run_time=3, rate_func=linear)
        self.wait(0.5)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(OVERLAP_COLOR)
        
        # Overlap highlighting updater for bars
        def bars_updater(m):
            n = n_tracker.get_value()
            for i, bar in enumerate(m):
                # If the bar index i is within the filter window relative to current n
                if abs(i - n) < 0.6:
                    bar.set_fill(color=OVERLAP_COLOR, opacity=0.9)
                else:
                    bar.set_fill(color=SIGNAL_COLOR, opacity=0.6)
        
        bars_f.add_updater(bars_updater)
        
        # Vertical indicator line for current output index n
        indicator = Line(ax_top.c2p(0, -0.2), ax_top.c2p(0, 2.8), color=OVERLAP_COLOR, stroke_width=2)
        indicator.add_updater(lambda l: l.move_to(ax_top.c2p(n_tracker.get_value(), 1.3)))
        
        self.add(indicator)
        n_tracker.set_value(0)
        self.play(n_tracker.animate.set_value(6), run_time=2, rate_func=linear)
        self.wait(0.5)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(RESULT_COLOR)
        
        self.add(ax_bottom, y_label)
        
        result_dots = VGroup()
        result_lines = VGroup()
        points = []

        # Point-by-point drawing of the smoother sequence
        n_tracker.set_value(-1)
        for i in range(len(signal_vals)):
            val = get_conv_val(i)
            p = ax_bottom.c2p(i, val)
            points.append(p)
            
            # Slide the window to index i
            self.play(n_tracker.animate.set_value(i), run_time=0.25, rate_func=linear)
            
            new_dot = Dot(p, color=RESULT_COLOR, radius=0.06)
            result_dots.add(new_dot)
            self.add(new_dot)
            
            if len(points) > 1:
                new_segment = Line(points[-2], points[-1], color=RESULT_COLOR, stroke_width=2)
                result_lines.add(new_segment)
                self.add(new_segment)
        
        self.wait(2)
