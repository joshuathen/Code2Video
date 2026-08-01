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
        # INITIAL SETUP
        title = "The Destination: Reaching π"
        lines = [
            "Our robot follows the circle's track.",
            "We measure the distance traveled along this arc.",
            "Walking for pi units takes us half-way around.",
            "We arrive exactly at the value negative one.",
            "This concludes the proof of Euler's famous identity."
        ]
        self.setup_layout(title, lines)

        # PREPARE OBJECTS
        # Complex Plane and Unit Circle
        complex_plane = ComplexPlane(
            x_range=[-2, 2, 1],
            y_range=[-2, 2, 1],
            x_length=4,
            y_length=4,
            background_line_style={"stroke_opacity": 0.4}
        )
        unit_circle = Circle(radius=complex_plane.get_x_unit_size(), color=BLUE_B)
        unit_circle.move_to(complex_plane.get_center())
        complex_plane_group = VGroup(complex_plane, unit_circle)
        
        # Use grid positioning to resolve Issue 50
        self.place_in_area(complex_plane_group, 'A2', 'D5', scale_factor=0.8)

        # Distance tracker and Dynamic label
        # Fixing the LaTeX error by using mob_class=Text for DecimalNumber
        val_tracker = ValueTracker(0)
        
        label_x_text = Text("x = ", font_size=32, color="#FFFF00")
        decimal_x = DecimalNumber(0, num_decimal_places=2, color="#FFFF00", mob_class=Text)
        decimal_x.add_updater(lambda d: d.set_value(val_tracker.get_value()))
        label_x_group = VGroup(label_x_text, decimal_x).arrange(RIGHT, buff=0.1)
        self.place_at_grid(label_x_group, 'A4', scale_factor=0.8)

        # Static labels for 1 and -1
        label_one = Text("1", font_size=36, color=WHITE)
        self.place_at_grid(label_one, 'C5', scale_factor=0.6)
        
        label_neg_one = Text("-1", font_size=36, color="#FF0000")
        # Precise grid positioning for Issue 51
        self.place_at_grid(label_neg_one, 'C2', scale_factor=0.6)

        # Formula - Use Text and literal pi character to avoid LaTeX dependency
        euler_identity = Text("e^iπ = -1", weight=BOLD, color="#FFFFFF")
        # Position using grid for Issue 52
        self.place_in_area(euler_identity, 'E2', 'F5', scale_factor=1.0)

        # === Animation for Lecture Line 1 ===
        # Line 1: "Our robot follows the circle's track."
        self.play(
            FadeIn(complex_plane_group),
            FadeIn(label_one),
            self.lecture[0].animate.set_color("#FFFF00"),
            run_time=1
        )
        self.wait(0.5)

        # === Animation for Lecture Line 2 ===
        # Line 2: "We measure the distance traveled along this arc."
        # Create persistent arc that updates based on the tracker
        arc_growth = Arc(
            radius=complex_plane.get_x_unit_size(),
            start_angle=0,
            angle=0.01,
            color="#FFFF00"
        )
        arc_growth.move_to(complex_plane.get_center())

        def arc_updater(m):
            angle = max(0.01, val_tracker.get_value())
            m.become(
                Arc(
                    radius=complex_plane.get_x_unit_size(),
                    start_angle=0,
                    angle=angle,
                    color="#FFFF00"
                ).move_to(complex_plane.get_center())
            )

        arc_growth.add_updater(arc_updater)

        self.play(
            FadeIn(label_x_group),
            Create(arc_growth),
            self.lecture[1].animate.set_color("#FFFF00"),
            run_time=1
        )

        # === Animation for Lecture Line 3 ===
        # Line 3: "Walking for pi units takes us half-way around."
        self.play(
            val_tracker.animate.set_value(PI),
            self.lecture[2].animate.set_color("#FFFF00"),
            run_time=3,
            rate_func=linear
        )
        self.wait(0.5)

        # === Animation for Lecture Line 4 ===
        # Line 4: "We arrive exactly at the value negative one."
        # Flash the destination point
        pos_neg_one = complex_plane.n2p(-1)
        self.play(
            Flash(pos_neg_one, color="#FF0000", flash_radius=0.4),
            FadeIn(label_neg_one),
            label_neg_one.animate.scale(1.3),
            self.lecture[3].animate.set_color("#FF0000"),
            run_time=1
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Line 5: "This concludes the proof of Euler's famous identity."
        self.play(
            FadeIn(euler_identity),
            self.lecture[4].animate.set_color("#FFFFFF"),
            run_time=1.5
        )
        self.wait(3)