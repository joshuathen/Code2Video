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
        
        # Labels for axes
        real_label = Text("Re", font_size=18, color=GREY_B)
        imag_label = Text("Im", font_size=18, color=GREY_B)
        
        # Fix for Issues 33, 34, 35: Place in area B2-F5 with scale 0.8 to avoid title and lecture notes
        plane_group = VGroup(plane, real_label, imag_label)
        self.place_in_area(plane_group, "B2", "F5", scale_factor=0.8)
        
        # Reposition labels relative to the moved plane
        real_label.next_to(plane.x_axis.get_end(), DOWN, buff=0.1)
        imag_label.next_to(plane.y_axis.get_end(), LEFT, buff=0.1)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(COLOR_LINE1)
        self.play(Create(plane_group))

        theta_tracker = ValueTracker(0)

        # Load asset: vector icon
        vector_icon = SVGMobject("/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/vector.svg")
        vector_icon.scale(0.15).set_color(COLOR_LINE1)
        
        # Elements initialized at theta=0
        origin = plane.c2p(0, 0)
        tip_init = plane.c2p(1, 0)
        pos_vec = Arrow(origin, tip_init, buff=0, color=COLOR_LINE1)
        
        # Velocity vector points straight up (perpendicular)
        v_end_init = plane.c2p(1, 0.7)
        vel_vec = Arrow(tip_init, v_end_init, buff=0, color=COLOR_LINE1)
        
        label_pos = Text("e^it", font_size=20, color=COLOR_LINE1)
        
        def update_moving_parts(group):
            t = theta_tracker.get_value()
            current_tip = plane.c2p(np.cos(t), np.sin(t))
            
            # 1. Update Position Vector
            pos_vec.put_start_and_end_on(origin, current_tip)
            
            # 2. Update Velocity Vector (perpendicular to position)
            # The direction in coordinate space is (-sin t, cos t)
            v_dir_coord = np.array([-np.sin(t), np.cos(t), 0])
            v_end_coord = np.array([np.cos(t), np.sin(t), 0]) + 0.7 * v_dir_coord
            v_end_px = plane.c2p(v_end_coord[0], v_end_coord[1])
            vel_vec.put_start_and_end_on(current_tip, v_end_px)
            
            # 3. Update Icon (offset from tip)
            pos_unit = (current_tip - origin) / np.linalg.norm(current_tip - origin)
            vector_icon.move_to(current_tip + 0.2 * pos_unit)
            
            # 4. Update Label
            label_pos.move_to(current_tip + 0.5 * pos_unit)

        moving_group = VGroup(pos_vec, vel_vec, vector_icon, label_pos)
        moving_group.add_updater(update_moving_parts)

        self.play(GrowArrow(pos_vec), FadeIn(vector_icon))
        self.play(GrowArrow(vel_vec), Write(label_pos))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(COLOR_LINE2)

        self.play(
            moving_group.animate.set_color(COLOR_LINE2),
            run_time=0.5
        )

        # Show growth forcing a turn
        self.play(theta_tracker.animate.set_value(PI/3), run_time=2, rate_func=linear)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(COLOR_LINE3)
        
        self.play(
            moving_group.animate.set_color(COLOR_LINE3),
            run_time=0.5
        )

        # Trace the full circular orbit
        start_arc = Arc(
            radius=plane.x_axis.get_unit_size(),
            start_angle=0,
            angle=PI/3,
            arc_center=origin,
            color=COLOR_LINE3
        )
        self.add(start_arc)

        remaining_arc = Arc(
            radius=plane.x_axis.get_unit_size(),
            start_angle=PI/3,
            angle=2*PI - PI/3,
            arc_center=origin,
            color=COLOR_LINE3
        )

        self.play(
            theta_tracker.animate.set_value(2*PI),
            Create(remaining_arc),
            run_time=4,
            rate_func=linear
        )
        self.wait(2)
