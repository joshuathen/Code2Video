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
        title_text = "Real-World Application: The Leaky Water Tank"
        lecture_lines = [
            "Consider water leaking from a tank at varying rates.",
            "The flow rate tells us the derivative of volume.",
            "Integrating that flow reveals the total water lost."
        ]
        self.setup_layout(title_text, lecture_lines)
        
        # Colors
        TANK_COLOR = "#4682B4"
        BUCKET_COLOR = "#C0C0C0"
        FLOW_COLOR = "#FF4500"
        WATER_COLOR = "#87CEEB"

        # === Animation for Lecture Line 1 ===
        # Show a blue water tank [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/tank.svg] (#4682B4) 
        # leaking into a silver bucket (#C0C0C0).
        
        # Resolve Issue 21: Load SVG asset
        tank_svg = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/tank.svg")
        tank_svg.set_color(TANK_COLOR)
        tank_label = Text("Water Tank", font_size=18, color=TANK_COLOR)
        tank_group = VGroup(tank_svg, tank_label).arrange(UP, buff=0.1)
        
        # Resolve Issue 31: Place in B1-C3 instead of A1-B3
        self.place_in_area(tank_group, "B1", "C3", scale_factor=0.8)
        
        bucket_body = RoundedRectangle(corner_radius=0.1, width=1.5, height=1.5, color=BUCKET_COLOR, fill_opacity=0.3)
        bucket_label = Text("Bucket", font_size=18, color=BUCKET_COLOR)
        bucket_group = VGroup(bucket_body, bucket_label).arrange(DOWN, buff=0.1)
        
        # Resolve Issue 33: Place in D1-E3 instead of E1-F3
        self.place_in_area(bucket_group, "D1", "E3", scale_factor=0.8)

        # Leak stream (connection between tank and bucket)
        stream = Line(tank_group.get_bottom(), bucket_group.get_top(), color=WATER_COLOR, stroke_width=4)
        
        self.lecture[0].set_color(TANK_COLOR)
        self.play(
            FadeIn(tank_group),
            FadeIn(bucket_group),
            Create(stream),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Plot a red 'Flow Rate' curve (#FF4500) on a side graph.
        
        axes = Axes(
            x_range=[0, 5, 1],
            y_range=[0, 2, 0.5],
            x_length=3.5,
            y_length=2.5,
            axis_config={"color": WHITE, "include_tip": True},
            tips=False
        )
        axes_labels = axes.get_axis_labels(x_label="t", y_label="f(t)")
        flow_graph_group = VGroup(axes, axes_labels)
        
        # Resolve Issue 32: scale_factor=0.8 instead of 0.9
        self.place_in_area(flow_graph_group, "B4", "E6", scale_factor=0.8)
        
        def flow_func(t):
            # A varying flow rate
            return 1 + 0.5 * np.sin(t)
            
        curve = axes.plot(flow_func, x_range=[0, 5], color=FLOW_COLOR)
        curve_label = Text("Flow Rate (f(t))", font_size=16, color=FLOW_COLOR)
        # Position label relative to the grid
        self.place_at_grid(curve_label, "A5", scale_factor=1.0)

        self.lecture[1].set_color(FLOW_COLOR)
        self.play(
            Create(axes),
            Create(axes_labels),
            run_time=1.5
        )
        self.play(
            Create(curve),
            FadeIn(curve_label),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Sync the bucket filling with the area under the flow graph.
        
        time_tracker = ValueTracker(0)
        
        # Area under curve (Integral)
        # Using a polygon update instead of always_redraw for consistency with instructions
        area = Polygon(
            axes.c2p(0, 0),
            axes.c2p(0, flow_func(0)),
            axes.c2p(0, 0),
            color=WATER_COLOR, 
            fill_opacity=0.4,
            stroke_width=0
        )
        
        def update_area(m):
            t = time_tracker.get_value()
            if t <= 0.01:
                return
            points = [axes.c2p(0, 0)]
            # Sample points for the curve area
            for x_val in np.linspace(0, t, 30):
                points.append(axes.c2p(x_val, flow_func(x_val)))
            points.append(axes.c2p(t, 0))
            m.set_points_as_corners(points)
            
        area.add_updater(update_area)
        
        # Integral of f(t) = 1 + 0.5 sin(t) is V(t) = t - 0.5 cos(t) + 0.5
        def integral_func(t):
            return t - 0.5 * np.cos(t) + 0.5
            
        max_integral = integral_func(5)
        
        # Water in the bucket
        bucket_water = Rectangle(
            width=bucket_body.width - 0.1,
            height=0.01,
            color=WATER_COLOR,
            fill_opacity=0.8,
            stroke_width=0
        )
        bucket_water.move_to(bucket_body.get_bottom(), aligned_edge=DOWN)
        
        # Update water level in bucket based on the integral
        def update_water(m):
            t = time_tracker.get_value()
            current_volume = integral_func(t)
            # Map volume to height
            h = (current_volume / max_integral) * (bucket_body.height - 0.1)
            m.stretch_to_fit_height(max(h, 0.001))
            m.move_to(bucket_body.get_bottom(), aligned_edge=DOWN)
            
        bucket_water.add_updater(update_water)
        
        self.lecture[2].set_color(WATER_COLOR)
        self.add(area, bucket_water)
        
        # Fill both area and bucket simultaneously
        self.play(
            time_tracker.animate.set_value(5),
            run_time=6,
            rate_func=linear
        )
        self.wait(2)
