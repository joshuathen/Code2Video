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

class Section2Scene(TeachingScene):
    def construct(self):
        # Title and Lecture Lines
        title = "Prerequisite Knowledge: Discrete vs. Continuous"
        lines = [
            "A histogram shows frequencies for discrete data.",
            "Thinner bars capture more precise measurements.",
            "As bars become microscopic, a smooth curve emerges."
        ]
        self.setup_layout(title, lines)
        
        # Colors - L008 (Hex strings)
        HIST_COLOR = "#FFCC80" # Light Orange
        CURVE_COLOR = "#FFFFE0" # Light Yellow
        AXES_COLOR = "#FFFFFF"
        
        # Define Gaussian-like function for visualization
        def gaussian(x):
            return np.exp(-x**2)
            
        # === Animation for Lecture Line 1 ===
        # Initial color highlight
        self.lecture[0].set_color(HIST_COLOR)
        
        # Define Axes (B2-F6 area - L002)
        # Resized slightly to fit within 4x4 grid area and leave room for labels
        axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[0, 1.2, 0.5],
            x_length=3.8,
            y_length=3.0,
            axis_config={"include_tip": False, "color": AXES_COLOR}
        )
        self.place_in_area(axes, "B2", "F6", scale_factor=1.0)
        self.add(axes)
        
        # 5 bars (Discrete representation)
        x_vals_5 = np.linspace(-2, 2, 5)
        # Width calculation based on axes scale (L025: avoid stretch)
        bar_width_5 = (axes.c2p(1, 0)[0] - axes.c2p(0, 0)[0]) * 0.8
        
        bars_5 = VGroup()
        for x in x_vals_5:
            h = gaussian(x)
            p_top = axes.c2p(x, h)
            p_bot = axes.c2p(x, 0)
            h_val = p_top[1] - p_bot[1]
            
            bar = Rectangle(
                width=bar_width_5, 
                height=h_val,
                fill_color=HIST_COLOR,
                fill_opacity=0.8,
                stroke_color=AXES_COLOR,
                stroke_width=1
            )
            bar.move_to(p_bot, aligned_edge=DOWN)
            bars_5.add(bar)
            
        discrete_label = Text("Discrete", font_size=20, color=HIST_COLOR)
        # Fix: Move to B2 per Issue 25 to avoid overlap with high bars/axes
        self.place_at_grid(discrete_label, "B2", scale_factor=0.8)
            
        self.play(Create(bars_5), FadeIn(discrete_label))
        self.wait(2.0)
        
        # === Animation for Lecture Line 2 ===
        # Transition highlight
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(HIST_COLOR)
        )
        
        # 50 bars (Increased precision)
        x_vals_50 = np.linspace(-2.5, 2.5, 50)
        bar_width_50 = (axes.c2p(0.1, 0)[0] - axes.c2p(0, 0)[0])
        
        bars_50 = VGroup()
        for x in x_vals_50:
            h = gaussian(x)
            p_top = axes.c2p(x, h)
            p_bot = axes.c2p(x, 0)
            h_val = max(p_top[1] - p_bot[1], 0.01) # Safety for non-zero height
            
            bar = Rectangle(
                width=bar_width_50,
                height=h_val,
                fill_color=HIST_COLOR,
                fill_opacity=0.6,
                stroke_width=0 # No stroke for dense bars (L002)
            )
            bar.move_to(p_bot, aligned_edge=DOWN)
            bars_50.add(bar)
            
        self.play(ReplacementTransform(bars_5, bars_50))
        self.wait(2.0)
        
        # === Animation for Lecture Line 3 ===
        # Final highlight
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(CURVE_COLOR)
        )
        
        # Bell Curve (Continuous limit)
        curve = axes.plot(lambda x: gaussian(x), color=CURVE_COLOR, x_range=[-3, 3])
        
        # Peak dashed line to trace height
        peak_line = DashedLine(
            start=axes.c2p(0, 0),
            end=axes.c2p(0, 1),
            color=CURVE_COLOR,
            stroke_width=2
        )
        
        continuous_label = Text("Continuous", font_size=20, color=CURVE_COLOR)
        # Fix: Move to B2 per Issue 26 to avoid overlap with peak
        self.place_at_grid(continuous_label, "B2", scale_factor=0.8)

        # Asset Integration: Bell Icon per Issue 20
        bell_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/bell.svg")
        bell_icon.set_fill(color=CURVE_COLOR, opacity=0.8) # Use set_fill for opacity (L031)
        self.place_at_grid(bell_icon, "B6", scale_factor=0.6)
        
        # Morphing transition - include the bell icon (Issue 20)
        self.play(
            ReplacementTransform(bars_50, curve),
            ReplacementTransform(discrete_label, continuous_label),
            FadeIn(bell_icon),
            run_time=2.5
        )
        self.play(Create(peak_line))
        
        # Final highlight using Indicate - L004
        self.play(Indicate(curve, color=CURVE_COLOR))
        self.play(Indicate(bell_icon, color=CURVE_COLOR))
        self.wait(2.0)
        
        # Final absorption time
        self.wait(2.0)
