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

class Section4Scene(TeachingScene):
    def construct(self):
        title_text = "The Fusion: Imaginary Growth is Circular"
        lecture_lines = [
            "Combine continuous growth with the rotation of i.",
            "Imaginary growth pushes sideways, perpendicular to the position.",
            "This constant sideways turn creates a perfect circular orbit."
        ]
        self.setup_layout(title_text, lecture_lines)

        # Colors
        COLOR_LINE1 = "#FFFFFF"
        COLOR_LINE2 = "#90EE90"
        COLOR_LINE3 = "#FFFFE0"

        # Plane Setup
        plane = Axes(
            x_range=[-1.5, 1.5, 1],
            y_range=[-1.5, 1.5, 1],
            x_length=4,
            y_length=4,
            axis_config={"include_tip": True, "color": GREY_D}
        )
        self.place_in_area(plane, "A1", "F6")

        # Labels for axes
        real_label = Text("Re", font_size=18, color=GREY_B).next_to(plane.x_axis.get_end(), DOWN, buff=0.1)
        imag_label = Text("Im", font_size=18, color=GREY_B).next_to(plane.y_axis.get_end(), LEFT, buff=0.1)
        plane_group = VGroup(plane, real_label, imag_label)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(COLOR_LINE1)
        self.play(Create(plane_group))

        theta_tracker = ValueTracker(0)

        # Persistent Position Vector
        pos_vec = Arrow(
            plane.c2p(0, 0),
            plane.c2p(1, 0),
            buff=0,
            color=COLOR_LINE1
        )
        def update_pos_vec(m):
            t = theta_tracker.get_value()
            m.put_start_and_end_on(
                plane.c2p(0, 0),
                plane.c2p(np.cos(t), np.sin(t))
            )
        pos_vec.add_updater(update_pos_vec)

        # Persistent Velocity Vector
        vel_vec = Arrow(
            plane.c2p(1, 0),
            plane.c2p(1, 0.7),
            buff=0,
            color=COLOR_LINE1
        )
        def update_vel_vec(m):
            t = theta_tracker.get_value()
            start = plane.c2p(np.cos(t), np.sin(t))
            end = plane.c2p(np.cos(t) - 0.7 * np.sin(t), np.sin(t) + 0.7 * np.cos(t))
            m.put_start_and_end_on(start, end)
        vel_vec.add_updater(update_vel_vec)

        # Persistent Label
        label_pos = Text("e^{it}", font_size=20, color=COLOR_LINE1)
        def update_label(m):
            t = theta_tracker.get_value()
            m.move_to(plane.c2p(1.2 * np.cos(t), 1.2 * np.sin(t)))
        label_pos.add_updater(update_label)

        self.play(GrowArrow(pos_vec))
        self.play(GrowArrow(vel_vec), Write(label_pos))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(COLOR_LINE2)

        # Update colors smoothly
        self.play(
            pos_vec.animate.set_color(COLOR_LINE2),
            vel_vec.animate.set_color(COLOR_LINE2),
            label_pos.animate.set_color(COLOR_LINE2)
        )

        self.play(theta_tracker.animate.set_value(PI/3), run_time=2, rate_func=linear)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(COLOR_LINE3)
        
        self.play(
            pos_vec.animate.set_color(COLOR_LINE3),
            vel_vec.animate.set_color(COLOR_LINE3),
            label_pos.animate.set_color(COLOR_LINE3)
        )

        # Orbit (Circular path)
        orbit = Arc(
            radius=plane.x_axis.get_unit_size(),
            start_angle=0,
            angle=0,
            arc_center=plane.c2p(0,0),
            color=COLOR_LINE3
        )
        def update_orbit(m):
            t = theta_tracker.get_value()
            m.become(Arc(
                radius=plane.x_axis.get_unit_size(),
                start_angle=0,
                angle=t,
                arc_center=plane.c2p(0,0),
                color=COLOR_LINE3
            ))
        orbit.add_updater(update_orbit)
        self.add(orbit)

        self.play(theta_tracker.animate.set_value(2*PI), run_time=4, rate_func=linear)
        self.wait(2)
