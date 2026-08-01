from manim import *
import math

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
        # Data
        title = "The Shape of Probability"
        lines = [
            "- Histograms visualize the probability of each success count.",
            "- Changing the probability p shifts the distribution's center.",
            "- Increasing the trial count n creates a bell shape."
        ]
        self.setup_layout(title, lines)
        
        # Colors
        COLOR_L1 = "#FFFF00"  # Yellow
        COLOR_L2 = "#00FFFF"  # Cyan
        COLOR_L3 = "#00FF00"  # Green
        AXES_COLOR = "#D3D3D3"
        
        # State trackers
        p_val = ValueTracker(0.3)
        n_val = ValueTracker(5)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(COLOR_L1)
        
        # Create Axes for histogram
        # x_range covers max n=20. y_range covers max prob (~0.6 for small n, p)
        axes = Axes(
            x_range=[0, 21, 5],
            y_range=[0, 0.6, 0.2],
            x_length=5,
            y_length=4,
            axis_config={"color": AXES_COLOR, "include_tip": False},
            tips=False
        )
        # Fix for Issue 34: self.place_in_area(axes, "B2", "E6", scale_factor=0.8)
        self.place_in_area(axes, "B2", "E6", scale_factor=0.8)
        
        k_label = Text("k (Successes)", font_size=16, color=AXES_COLOR)
        pk_label = Text("P(k)", font_size=16, color=AXES_COLOR)
        k_label.next_to(axes.x_axis, DOWN, buff=0.2)
        pk_label.next_to(axes.y_axis, LEFT, buff=0.2)
        
        # Pre-create bars (21 for max n=20) to maintain persistence
        bars = VGroup()
        for k in range(21):
            bar = Rectangle(
                width=axes.x_axis.get_unit_size() * 0.8,
                height=0.01,
                fill_opacity=0.7,
                fill_color=COLOR_L1,
                stroke_width=1,
                stroke_color=WHITE
            )
            bar.move_to(axes.c2p(k, 0), aligned_edge=DOWN)
            bars.add(bar)
            
        def update_bars(m):
            n_curr = int(n_val.get_value())
            p_curr = p_val.get_value()
            unit_h = axes.y_axis.get_unit_size()
            current_color = m.get_color()
            for k, bar in enumerate(m):
                if k <= n_curr:
                    try:
                        # Binomial Probability calculation: P(k) = C(n,k) * p^k * (1-p)^(n-k)
                        prob = math.comb(n_curr, k) * (p_curr**k) * ((1-p_curr)**(n_curr-k))
                    except (ValueError, OverflowError):
                        prob = 0
                    
                    new_height = max(unit_h * prob, 0.05) # Minimum height for visibility
                    bar.stretch_to_fit_height(new_height, about_edge=DOWN)
                    bar.move_to(axes.c2p(k, 0), aligned_edge=DOWN)
                    bar.set_fill(color=current_color, opacity=0.7)
                    bar.set_stroke(opacity=1)
                else:
                    bar.set_fill(opacity=0)
                    bar.set_stroke(opacity=0)

        # Initialize bars state
        update_bars(bars)
        
        self.play(Create(axes), Write(k_label), Write(pk_label))
        self.play(FadeIn(bars))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(COLOR_L2)
        bars.set_color(COLOR_L2)
        
        # Controls readouts
        p_label_text = Text("p = ", font_size=20, color=COLOR_L2)
        p_num = DecimalNumber(p_val.get_value(), num_decimal_places=2, color=COLOR_L2, font_size=20)
        p_group = VGroup(p_label_text, p_num).arrange(RIGHT, buff=0.1)
        
        n_label_text = Text("n = ", font_size=20, color=WHITE)
        n_num = DecimalNumber(n_val.get_value(), num_decimal_places=0, color=WHITE, font_size=20)
        n_group = VGroup(n_label_text, n_num).arrange(RIGHT, buff=0.1)
        
        controls = VGroup(p_group, n_group).arrange(RIGHT, buff=1.0)
        # Fix for Issue 35: self.place_in_area(controls, "A2", "A6", scale_factor=0.8)
        self.place_in_area(controls, "A2", "A6", scale_factor=0.8)
        
        # Updaters for real-time changes
        p_num.add_updater(lambda d: d.set_value(p_val.get_value()))
        bars.add_updater(update_bars)
        
        self.play(FadeIn(controls))
        # Animate p slider behavior
        self.play(p_val.animate.set_value(0.9), run_time=2.5, rate_func=linear)
        self.play(p_val.animate.set_value(0.1), run_time=2.5, rate_func=linear)
        self.play(p_val.animate.set_value(0.5), run_time=1.5, rate_func=linear)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(COLOR_L3)
        bars.set_color(COLOR_L3)
        
        # Highlight n readout
        n_label_text.set_color(COLOR_L3)
        n_num.set_color(COLOR_L3)
        
        n_num.add_updater(lambda d: d.set_value(n_val.get_value()))
        
        # Animate n increasing to show smoothing effect
        self.play(n_val.animate.set_value(20), run_time=5, rate_func=smooth)
        self.wait(2)
        
        # Final cleanup
        bars.remove_updater(update_bars)
        p_num.remove_updater(None)
        n_num.remove_updater(None)
