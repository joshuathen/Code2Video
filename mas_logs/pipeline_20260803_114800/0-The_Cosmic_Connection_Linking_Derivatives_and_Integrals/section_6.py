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

class Section6Scene(TeachingScene):
    def construct(self):
        # 1. Setup Layout & Script
        # Lecture lines for Section 6
        lecture_lines = [
            "Flow rate is the derivative of volume.",
            "Volume is the integral of flow rate.",
            "Calculus links these two quantities in real-time."
        ]
        self.setup_layout("Real-World Application: The Water Tank", lecture_lines)

        # 2. Asset Integration (Issue 22)
        tank_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/tank.svg"
        meter_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/meter.svg"
        
        tank_asset = SVGMobject(tank_path).set_color("#1E90FF")
        meter_asset = SVGMobject(meter_path).set_color(WHITE)
        
        # Issue 39: Positioning the tank asset
        self.place_at_grid(tank_asset, 'B5', scale_factor=1.1)
        
        # Meter and Label placement relative to tank
        meter_asset.scale(0.3).next_to(tank_asset, LEFT, buff=0.1).shift(UP * 0.2)
        meter_label = Text("L/s", font_size=14, color=WHITE).next_to(meter_asset, UP, buff=0.05)
        
        # Physical indicators (Needle for meter, Water for tank)
        needle = Line(meter_asset.get_center(), meter_asset.get_center() + UP * 0.15, color=RED, stroke_width=3)
        
        water = Rectangle(
            width=tank_asset.width * 0.82,
            height=0.01,
            fill_color="#1E90FF",
            fill_opacity=0.6,
            stroke_width=0
        ).move_to(tank_asset.get_bottom() + UP * 0.15, aligned_edge=DOWN)
        
        # 3. Mathematical & Graphical Elements
        # Issue 37: Mathematical formula placement
        calc_formula = MathTex(
            r"V(t) = \int_0^t Q(\tau) \, d\tau", 
            font_size=32, 
            color=WHITE
        )
        self.place_in_area(calc_formula, 'A1', 'B3', scale_factor=0.8)

        # Issue 38: Graphical representation placement
        axes = Axes(
            x_range=[0, 5, 1],
            y_range=[0, 2, 0.5],
            x_length=4.0,
            y_length=2.2,
            axis_config={"include_tip": False, "font_size": 18, "color": GREY},
            tips=False
        ).add_coordinates()
        
        # Define flow rate function Q(t) = 1 + 0.5*sin(t)
        flow_rate_func = lambda x: 1 + 0.5 * np.sin(x)
        flow_graph = axes.plot(flow_rate_func, x_range=[0, 5], color=YELLOW)
        flow_label = axes.get_graph_label(flow_graph, label="Q(t)", x_val=4.5, direction=UP)
        
        tank_graph = VGroup(axes, flow_graph, flow_label)
        self.place_in_area(tank_graph, 'C2', 'F6', scale_factor=0.85)

        # 4. Dynamic Logic (ValueTracker & Updaters)
        time_tracker = ValueTracker(0)
        
        # Integral of Q(t) = 1 + 0.5*sin(t) from 0 to T is T - 0.5*cos(T) + 0.5
        def get_volume(t):
            return t - 0.5 * np.cos(t) + 0.5
            
        max_vol_at_5 = get_volume(5)
        max_tank_h = tank_asset.height * 0.75
        
        # Water level updater
        water.add_updater(lambda w: w.stretch_to_fit_height(
            max(0.01, (get_volume(time_tracker.get_value()) / max_vol_at_5) * max_tank_h),
            about_edge=DOWN
        ))
        
        # Meter needle updater (Issue 22: meter fluctuations)
        def update_needle(n):
            rate = flow_rate_func(time_tracker.get_value())
            # Map rate [0.5, 1.5] to polar angles [135 deg to 45 deg]
            angle = interpolate(3 * PI / 4, PI / 4, (rate - 0.5) / 1.0)
            start_point = meter_asset.get_center()
            end_point = start_point + 0.15 * np.array([np.cos(angle), np.sin(angle), 0])
            n.put_start_and_end_on(start_point, end_point)
            
        needle.add_updater(update_needle)
        
        # Counter updater (Total Liters)
        total_liters_label = Text("Total Volume (L):", font_size=20, color="#00FF00")
        counter = DecimalNumber(0, font_size=24, color="#00FF00", num_decimal_places=1)
        counter_group = VGroup(total_liters_label, counter).arrange(RIGHT, buff=0.2)
        counter_group.next_to(tank_asset, DOWN, buff=0.3)
        
        counter.add_updater(lambda c: c.set_value(get_volume(time_tracker.get_value()) * 10))
        
        # Moving dot on graph
        dot = Dot(color=YELLOW).scale(0.8)
        dot.add_updater(lambda d: d.move_to(axes.c2p(time_tracker.get_value(), flow_rate_func(time_tracker.get_value()))))
        
        # Cumulative area under curve (Integral visualization)
        area = always_redraw(lambda: axes.get_area(
            flow_graph, 
            x_range=[0, time_tracker.get_value()], 
            color=YELLOW, 
            opacity=0.3
        ))

        # === Animation for Lecture Line 1 ===
        # "Flow rate is the derivative of volume."
        self.play(self.lecture[0].animate.set_color(BLUE))
        self.play(
            FadeIn(tank_asset),
            FadeIn(water),
            FadeIn(meter_asset),
            FadeIn(meter_label),
            FadeIn(needle),
            Create(axes),
            Write(calc_formula)
        )
        self.play(Create(flow_graph), FadeIn(flow_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "Volume is the integral of flow rate."
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(BLUE)
        )
        self.play(FadeIn(counter_group), FadeIn(dot))
        self.add(area)
        
        # Animate the process (Calculus in real-time)
        self.play(
            time_tracker.animate.set_value(5),
            run_time=8,
            rate_func=linear
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "Calculus links these two quantities in real-time."
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(BLUE)
        )
        
        # Visual highlight of the link
        connection_box = SurroundingRectangle(VGroup(counter_group, meter_asset), color=BLUE, buff=0.1)
        self.play(Create(connection_box))
        self.play(FadeOut(connection_box))
        self.wait(2)

        # Final cleanup for potential transitions
        self.play(
            *[FadeOut(m) for m in [tank_asset, water, meter_asset, meter_label, needle, tank_graph, calc_formula, counter_group, dot, area]]
        )
