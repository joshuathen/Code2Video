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

class Section3Scene(TeachingScene):
    def construct(self):
        title_text = "The Math of Shading: Integration"
        lecture_lines = [
            "Calculus calculates these areas through integration.",
            "We integrate the density function over an interval.",
            "Shaded regions represent the probability for that range.",
            "As the interval narrows, the area approaches zero.",
            "This confirms why exact point probabilities are zero."
        ]
        self.setup_layout(title_text, lecture_lines)

        # === Animation for Lecture Line 1 ===
        # Display integral notation ∫ f(x) dx #FFFF00 from a to b.
        self.lecture[0].set_color("#FFFF00")
        integral_formula = MathTex(r"P(a \le X \le b) = \int_{a}^{b} f(x) dx", color="#FFFF00")
        # Issue 26 Fix: Move to row A
        self.place_in_area(integral_formula, "A2", "A5", scale_factor=0.9)
        self.play(Write(integral_formula))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # We integrate the density function over an interval.
        self.lecture[1].set_color("#FFFF00")
        
        # Setup Axes and Graph
        axes = Axes(
            x_range=[0, 4, 1],
            y_range=[0, 1, 0.5],
            axis_config={"include_tip": False},
            x_length=4,
            y_length=3
        ).scale(0.7)
        self.place_in_area(axes, "C2", "E5")
        
        def pdf_func(x):
            # A simple bell-shaped curve for visualization
            return 0.8 * np.exp(-(x - 2)**2)
            
        graph = axes.plot(pdf_func, color=WHITE)
        
        # Values for a and b
        a_val = 1.2
        b_start = 2.8
        b_tracker = ValueTracker(b_start)
        
        # Static part: point a
        line_a = axes.get_vertical_line(axes.input_to_graph_point(a_val, graph), color=WHITE)
        label_a = MathTex("a", font_size=24).next_to(axes.c2p(a_val, 0), DOWN)
        
        # Dynamic part: point b
        line_b = Line(axes.c2p(b_start, 0), axes.input_to_graph_point(b_start, graph), color=WHITE)
        line_b.add_updater(lambda m: m.put_start_and_end_on(
            axes.c2p(b_tracker.get_value(), 0),
            axes.input_to_graph_point(b_tracker.get_value(), graph)
        ))
        
        label_b = MathTex("b", font_size=24)
        label_b.add_updater(lambda m: m.next_to(axes.c2p(b_tracker.get_value(), 0), DOWN))

        self.play(Create(axes), Create(graph))
        self.play(Create(line_a), Write(label_a), Create(line_b), Write(label_b))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Shaded regions represent the probability for that range.
        self.lecture[2].set_color("#00FFFF")
        
        # Initial area #00FFFF
        # L011: Introduce only at designated step
        area = always_redraw(lambda: axes.get_area(
            graph, 
            x_range=[a_val, b_tracker.get_value()], 
            color="#00FFFF", 
            opacity=0.5
        ))
        
        self.play(FadeIn(area))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # As the interval narrows, the area approaches zero.
        self.lecture[3].set_color("#00FFFF")
        
        # Move b toward a until they overlap
        self.play(b_tracker.animate.set_value(a_val), run_time=3)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # This confirms why exact point probabilities are zero.
        # Display 'P(X=a) = 0' #FF00FF as the lines merge.
        self.lecture[4].set_color("#FF00FF")
        
        prob_zero = MathTex("P(X=a) = 0", color="#FF00FF")
        # Issue 24 Fix: Move to E6 to avoid overlap
        self.place_at_grid(prob_zero, "E6", scale_factor=0.8)
        
        self.play(Write(prob_zero))
        self.play(Indicate(prob_zero)) # L004
        self.wait(2)
        
        # Scanner bar [Asset: ...] #FFFFFF moves from a to b, filling area.
        # Issue 17: Asset integration
        scanner_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/scanner.svg")
        scanner_asset.set_color("#FFFFFF")
        scanner_asset.scale(0.3)
        
        self.play(FadeOut(prob_zero), FadeOut(area))
        
        # Reset b for range visualization
        self.play(b_tracker.animate.set_value(b_start), run_time=0.5)
        
        scan_x = ValueTracker(a_val)
        
        # The physical bar
        scanner_line = Line(
            axes.c2p(a_val, 0),
            axes.input_to_graph_point(a_val, graph),
            color="#FFFFFF",
            stroke_width=4
        )
        scanner_line.add_updater(lambda m: m.put_start_and_end_on(
            axes.c2p(scan_x.get_value(), 0),
            axes.input_to_graph_point(scan_x.get_value(), graph)
        ))
        
        # The icon (Asset)
        scanner_asset.add_updater(lambda m: m.move_to(
            axes.input_to_graph_point(scan_x.get_value(), graph) + UP * 0.3
        ))
        
        filling_area = always_redraw(lambda: axes.get_area(
            graph,
            x_range=[a_val, scan_x.get_value()],
            color="#00FFFF",
            opacity=0.5
        ))
        
        # Counter for probability
        counter = DecimalNumber(0, num_decimal_places=3, color=WHITE)
        counter.add_updater(lambda d: d.set_value(
            self.calc_prob(a_val, scan_x.get_value())
        ))
        counter_label = Text("Area:", font_size=20, color=WHITE)
        counter_group = VGroup(counter_label, counter).arrange(RIGHT, buff=0.2)
        # Issue 25 Fix: Move to B6
        self.place_at_grid(counter_group, "B6", scale_factor=0.8)
        
        self.add(filling_area, scanner_line, scanner_asset, counter_group)
        self.play(scan_x.animate.set_value(b_start), run_time=4, rate_func=linear)
        self.wait(3)

    def calc_prob(self, start, end):
        # Numerical integration for visual counter
        if abs(start - end) < 1e-4: return 0.0
        steps = 20
        xs = np.linspace(start, end, steps)
        ys = 0.8 * np.exp(-(xs - 2)**2)
        # Approximate area under the curve
        return np.trapz(ys, xs)
