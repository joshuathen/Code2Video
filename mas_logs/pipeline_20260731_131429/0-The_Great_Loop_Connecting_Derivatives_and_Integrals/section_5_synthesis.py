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

class Section5SynthesisScene(TeachingScene):
    def construct(self):
        # Initial Setup
        title = "Visual Synthesis & Animation"
        lecture_lines = [
            "The slope of the area graph matches the height.",
            "Watch the rate of change and accumulation unite.",
            "The Great Loop of Calculus is now complete."
        ]
        self.setup_layout(title, lecture_lines)

        # Colors from Storyboard
        BALLOON_COLOR = "#FF69B4"
        GAUGE_COLOR = "#00FF00"
        CURVE_COLOR = "#FFB6C1"
        
        # Tracker for synchronization (ranges from 0 to 1)
        time_tracker = ValueTracker(0)

        # === Animation for Lecture Line 1 ===
        # "The slope of the area graph matches the height."
        self.lecture[0].set_color(GAUGE_COLOR)

        # 1. Gauge (Air Flow Rate) [Asset: gauge.svg]
        # Resolved Issue 34: Moving gauge to A2
        gauge_svg = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/gauge.svg")
        gauge_svg.set_color(GAUGE_COLOR)
        self.place_at_grid(gauge_svg, 'A2', scale_factor=0.6)
        
        gauge_label = Text("Air Flow Rate", font_size=16, color=GAUGE_COLOR).next_to(gauge_svg, UP, buff=0.1)
        
        # Adding a visual bar indicator for the flow value next to the gauge
        gauge_bar_bg = Rectangle(width=1.5, height=0.2, color=GRAY, fill_opacity=0.3).next_to(gauge_svg, DOWN, buff=0.1)
        gauge_bar = Rectangle(width=0.01, height=0.2, color=GAUGE_COLOR, fill_opacity=0.8, stroke_width=0)
        gauge_bar.align_to(gauge_bar_bg, LEFT)
        
        def update_gauge(m):
            t = time_tracker.get_value()
            # Flow rate function f(t) = sin(pi * t)
            flow_val = np.sin(np.pi * t)
            new_width = max(0.01, flow_val * 1.5)
            m.stretch_to_fit_width(new_width)
            m.align_to(gauge_bar_bg, LEFT)

        gauge_bar.add_updater(update_gauge)

        # 2. Balloon Character [Asset: balloon.svg]
        # Resolved Issue 35: Shrinking area to B1-D2
        balloon_svg = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/balloon.svg")
        balloon_svg.set_color(BALLOON_COLOR)
        self.place_in_area(balloon_svg, 'B1', 'D2', scale_factor=0.8)
        
        # Center point for balloon anchoring (calculated for B1-D2 area: mid of (0.5, 1.2) and (1.5, -0.8))
        balloon_center = np.array([1.0, 0.2, 0])
        
        def update_balloon(m):
            t = time_tracker.get_value()
            # Volume V(t) = (1/pi) * (1 - cos(pi * t))
            volume = (1/np.pi) * (1 - np.cos(np.pi * t))
            # Base width + growth
            growth_width = 0.5 + 2.0 * np.sqrt(volume)
            m.stretch_to_fit_width(growth_width)
            m.stretch_to_fit_height(growth_width) 
            m.move_to(balloon_center)
        
        balloon_svg.add_updater(update_balloon)
        
        self.play(FadeIn(gauge_svg), FadeIn(gauge_label), FadeIn(gauge_bar_bg), FadeIn(balloon_svg))
        self.add(gauge_bar)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "Watch the rate of change and accumulation unite."
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(CURVE_COLOR)

        # 3. Volume Curve Graph - Visualizes the Integral
        axes = Axes(
            x_range=[0, 1.2, 0.5],
            y_range=[0, 0.8, 0.2],
            x_length=3.0,
            y_length=4.0,
            axis_config={"include_tip": True, "font_size": 16}
        )
        # Position axes on the right side
        self.place_in_area(axes, "A4", "F6", scale_factor=0.9)
        
        graph_label = Text("Volume (Integral)", font_size=18, color=CURVE_COLOR).next_to(axes, UP, buff=0.2)
        x_label = Text("Time", font_size=14).next_to(axes.x_axis, RIGHT, buff=0.1)
        
        # Dynamic curve plotting
        volume_curve = always_redraw(lambda: axes.plot(
            lambda t: (1/np.pi) * (1 - np.cos(np.pi * t)),
            x_range=[0, max(0.0001, time_tracker.get_value())],
            color=CURVE_COLOR
        ))

        self.play(Create(axes), FadeIn(graph_label), FadeIn(x_label))
        self.add(volume_curve)
        
        # Inflation and Curve Drawing
        self.play(time_tracker.animate.set_value(1.0), run_time=5, rate_func=linear)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "The Great Loop of Calculus is now complete."
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)

        # 4. Connecting Slope to Rate
        t_demo = 0.5
        v_demo = (1/np.pi) * (1 - np.cos(np.pi * t_demo))
        
        # Point on the Volume curve
        dot = Dot(axes.c2p(t_demo, v_demo), color=WHITE)
        
        # Manual Tangent Line: At t=0.5, V'(t) = 1.0
        p1 = axes.c2p(t_demo - 0.25, v_demo - 0.25)
        p2 = axes.c2p(t_demo + 0.25, v_demo + 0.25)
        tangent = Line(p1, p2, color=GAUGE_COLOR, stroke_width=6)
        
        # Resolved Issue 33: Moving slope_info to A4 and scaling down
        slope_info = Text("Slope = Flow Rate", font_size=18, color=GAUGE_COLOR)
        self.place_at_grid(slope_info, 'A4', scale_factor=0.6)

        # Connection Arrow
        connection_arrow = Arrow(
            start=gauge_svg.get_center(),
            end=tangent.get_center(),
            color=GAUGE_COLOR,
            stroke_width=4,
            buff=0.3
        )

        # Move tracker back to demo point
        self.play(time_tracker.animate.set_value(t_demo), run_time=1.5)
        self.play(Create(dot), Create(tangent))
        self.play(FadeIn(slope_info), GrowArrow(connection_arrow))
        
        self.wait(3)
