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
        title_text = "Application: The Arrival Window"
        lecture_lines = [
            "Imagine Zippy the drone delivering your package today.",
            "The curve shows likely arrival times around ten-thirty.",
            "Shading the region from ten-twenty-five to ten-thirty-five.",
            "This area represents the chance of an on-time delivery.",
            "Calculus helps us calculate this specific shaded area."
        ]
        self.setup_layout(title_text, lecture_lines)

        # Colors from storyboard
        DRONE_COLOR = "#87CEEB"
        CURVE_COLOR = "#AFEEEE"
        SHADE_COLOR = "#FFFACD"
        TEXT_COLOR = "#FFFFFF"
        INTEGRAL_COLOR = "#FFD700"

        # === Animation for Lecture Line 1 ===
        # Show the drone asset [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/drone.svg]
        # and a timeline
        drone = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/drone.svg").set_color(DRONE_COLOR)
        self.place_at_grid(drone, "A3", scale_factor=0.6)
        
        timeline = NumberLine(
            x_range=[10, 11, 0.5],
            length=6,
            include_numbers=False,
            include_tip=True,
            color=GRAY
        )
        t_1000 = Text("10:00", font_size=16).next_to(timeline.n2p(10), DOWN)
        t_1030 = Text("10:30", font_size=16).next_to(timeline.n2p(10.5), DOWN)
        t_1100 = Text("11:00", font_size=16).next_to(timeline.n2p(11), DOWN)
        timeline_group = VGroup(timeline, t_1000, t_1030, t_1100)
        self.place_in_area(timeline_group, "E1", "E6", scale_factor=0.8)

        # Hovering effect
        drone.starting_y = drone.get_center()[1]
        def drone_hover(m):
            m.set_y(drone.starting_y + 0.1 * np.sin(self.time * 2))
        drone.add_updater(drone_hover)

        self.play(
            self.lecture[0].animate.set_color(DRONE_COLOR),
            FadeIn(drone),
            Create(timeline_group)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Draw a bell curve centered at a "10:30" marker.
        # Fix 35: self.place_in_area(graph_elements, 'B1', 'E6', scale_factor=0.8)
        axes = Axes(
            x_range=[10, 11, 0.1],
            y_range=[0, 1.2, 0.4],
            x_length=6,
            y_length=3,
            axis_config={"include_tip": False}
        ).set_color(GRAY)
        
        # Labels for axes
        a_1000 = Text("10:00", font_size=16).next_to(axes.c2p(10, 0), DOWN)
        a_1030 = Text("10:30", font_size=16).next_to(axes.c2p(10.5, 0), DOWN)
        a_1100 = Text("11:00", font_size=16).next_to(axes.c2p(11, 0), DOWN)
        axes_labels = VGroup(a_1000, a_1030, a_1100)
        
        # Normal distribution curve
        mu = 10.5
        sigma = 0.12
        curve = axes.plot(
            lambda x: np.exp(-0.5 * ((x - mu) / sigma)**2) / (sigma * np.sqrt(2 * np.pi)) * 0.3,
            color=CURVE_COLOR,
            x_range=[10, 11]
        )
        
        graph_elements = VGroup(axes, axes_labels, curve)
        self.place_in_area(graph_elements, 'B1', 'E6', scale_factor=0.8)

        self.play(
            self.lecture[1].animate.set_color(CURVE_COLOR),
            FadeOut(timeline_group),
            FadeIn(VGroup(axes, axes_labels)),
            Create(curve)
        )
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # Shading the region from 10:25 to 10:35
        # 10:25 = 10 + 25/60 = 10.4166...
        # 10:35 = 10 + 35/60 = 10.5833...
        start_x = 10 + 25/60
        end_x = 10 + 35/60
        area = axes.get_area(curve, x_range=[start_x, end_x], color=SHADE_COLOR, opacity=0.6)
        
        self.play(
            self.lecture[2].animate.set_color(SHADE_COLOR),
            FadeIn(area)
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Pulse area and display text
        # Fix 36: self.place_in_area(label, 'A2', 'A5', scale_factor=0.7)
        label_text = Text("Probability of Arrival Window", font_size=20, color=TEXT_COLOR)
        self.place_in_area(label_text, 'A2', 'A5', scale_factor=0.7)

        # Remove updater to avoid conflict with animation, move drone to A1
        drone.remove_updater(drone_hover)
        
        self.play(
            self.lecture[3].animate.set_color(TEXT_COLOR),
            Write(label_text),
            area.animate.set_fill(opacity=0.9),
            drone.animate.move_to(self.grid["A1"]),
            run_time=1
        )
        
        # Resume hover at new location
        drone.starting_y = drone.get_center()[1]
        drone.add_updater(drone_hover)
        
        self.play(area.animate.set_fill(opacity=0.6), run_time=1)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Fix 34: self.place_in_area(integral_math, 'F1', 'F6', scale_factor=0.8)
        # An integration symbol slides over the shaded area
        integral_math = MathTex(
            r"P(10:25 \le X \le 10:35) = \int_{10:25}^{10:35} f(x) \, dx",
            font_size=24,
            color=INTEGRAL_COLOR
        )
        self.place_in_area(integral_math, 'F1', 'F6', scale_factor=0.8)
        
        # Start integral_math at area center and slide to target
        target_pos = integral_math.get_center()
        integral_math.move_to(area.get_center()).set_opacity(0)

        self.play(
            self.lecture[4].animate.set_color(INTEGRAL_COLOR),
            integral_math.animate.move_to(target_pos).set_opacity(1),
            run_time=2
        )
        self.wait(3)

        # Clean up
        drone.remove_updater(drone_hover)
