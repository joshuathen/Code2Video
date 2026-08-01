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
        # Fetching data from storyboard
        title_text = "Visualizing the Distribution Shape"
        lecture_lines = [
            "Probability mass functions show the distribution of results.",
            "When p is low, the graph skews right.",
            "As p increases, the peak shifts to the right.",
            "High trial counts create a symmetric bell curve.",
            "This shape represents the most likely outcomes."
        ]
        
        self.setup_layout(title_text, lecture_lines)
        
        # Define constants and trackers
        chart_color = "#ADD8E6"
        highlight_color = YELLOW
        n_tracker = ValueTracker(10)
        p_tracker = ValueTracker(0.5)
        
        # Create Axes
        axes = Axes(
            x_range=[0, 31, 5],
            y_range=[0, 0.5, 0.1],
            x_length=5,
            y_length=4,
            axis_config={"include_tip": False}
        ).add_coordinates(font_size=14)
        # Resolved Issue 36: Adjusted area and scale factor
        self.place_in_area(axes, 'B1', 'F6', scale_factor=0.7)
        
        # Axis Labels
        x_label = Text("Number of Successes (k)", font_size=16, color=WHITE)
        x_label.next_to(axes.x_axis, DOWN, buff=0.4)
        y_label = Text("P(X=k)", font_size=16, color=WHITE).rotate(90*DEGREES)
        y_label.next_to(axes.y_axis, LEFT, buff=0.4)
        
        self.add(axes, x_label, y_label)

        # Create Bars
        bars = VGroup()
        for i in range(31):
            bar = Rectangle(
                width=0.12,
                height=0.01,
                fill_opacity=0.8,
                fill_color=chart_color,
                stroke_width=0.5,
                stroke_color=WHITE
            )
            bar.move_to(axes.c2p(i, 0), aligned_edge=DOWN)
            bars.add(bar)
            
        def update_bars(m):
            n = int(n_tracker.get_value())
            p = p_tracker.get_value()
            for i, b in enumerate(m):
                if i <= n:
                    # Binomial PMF: (n choose i) * p^i * (1-p)^(n-i)
                    try:
                        # math.comb is safe for n <= 30
                        prob = math.comb(n, i) * (p**i) * ((1-p)**(n-i))
                    except:
                        prob = 0
                    
                    # Convert probability to visual height
                    target_height = axes.c2p(0, prob)[1] - axes.c2p(0, 0)[1]
                    # Update height using stretch_to_fit_height
                    b.stretch_to_fit_height(max(target_height, 0.001))
                    b.move_to(axes.c2p(i, 0), aligned_edge=DOWN)
                    b.set_opacity(0.8)
                else:
                    b.set_opacity(0)
                    
        bars.add_updater(update_bars)
        self.add(bars)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(highlight_color))
        self.wait(2)

        # === Animation for Lecture Line 2 ===
        skew_right_label = Text("Skewed Right", font_size=20, color=WHITE)
        # Resolved Issue 34: Adjusted area and scale factor
        self.place_in_area(skew_right_label, 'A4', 'A6', scale_factor=0.8)
        
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(highlight_color),
            p_tracker.animate.set_value(0.2),
            Write(skew_right_label),
            run_time=2
        )
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        skew_left_label = Text("Skewed Left", font_size=20, color=WHITE)
        # Resolved Issue 35: Adjusted area and scale factor
        self.place_in_area(skew_left_label, 'A1', 'A3', scale_factor=0.8)

        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(highlight_color),
            p_tracker.animate.set_value(0.8),
            ReplacementTransform(skew_right_label, skew_left_label),
            run_time=2
        )
        self.wait(2)

        # === Animation for Lecture Line 4 ===
        self.play(
            self.lecture[2].animate.set_color(WHITE),
            self.lecture[3].animate.set_color(highlight_color),
            p_tracker.animate.set_value(0.5),
            n_tracker.animate.set_value(30),
            FadeOut(skew_left_label),
            run_time=2
        )
        self.wait(2)

        # === Animation for Lecture Line 5 ===
        # For n=30, p=0.5, the mode is 15.
        highest_bar = bars[15]
        
        self.play(
            self.lecture[3].animate.set_color(WHITE),
            self.lecture[4].animate.set_color(highlight_color),
        )
        self.play(Flash(highest_bar, color=YELLOW, flash_radius=0.5, line_length=0.3))
        self.wait(2)
        
        # End state
        self.play(self.lecture[4].animate.set_color(WHITE))
        self.wait(1)
