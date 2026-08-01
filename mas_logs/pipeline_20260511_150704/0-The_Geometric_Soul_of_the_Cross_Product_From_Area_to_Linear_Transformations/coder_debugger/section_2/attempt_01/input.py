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
        # 1. Setup Layout
        title = "The Geometry of Magnitude: Parallelograms"
        lines = [
            "In 3D, the cross product's magnitude represents area.", 
            "This area equals magnitude u times v times sine theta.", 
            "As the angle grows, the solar panel's area increases."
        ]
        self.setup_layout(title, lines)
        
        # Colors
        color_u = "#52CEFF"
        color_v = "#C6FF00"
        color_area = "#FFD700"
        highlight_color = YELLOW

        # Shared trackers
        angle_tracker = ValueTracker(30 * DEGREES)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(highlight_color)

        # Build Parallelogram components relative to ORIGIN for grouping
        u_vec = Arrow(ORIGIN, [3.0, 0, 0], buff=0, color=color_u, stroke_width=4)
        v_vec = Arrow(ORIGIN, [2.16, 1.25, 0], buff=0, color=color_v, stroke_width=4)
        area_poly = Polygon(ORIGIN, [3.0, 0, 0], [5.16, 1.25, 0], [2.16, 1.25, 0], 
                            fill_opacity=0.4, fill_color=color_area, stroke_width=1, stroke_color=color_area)
        
        # Group and position (Issue 36 Fix)
        parallelogram_group = VGroup(u_vec, v_vec, area_poly)
        self.place_in_area(parallelogram_group, 'B2', 'E6', scale_factor=0.8)
        
        # Extract dynamic anchors after positioning
        origin_pos = u_vec.get_start()
        v_len = np.linalg.norm(v_vec.get_vector())

        # Vector labels (Issue 37 Fix for label_u)
        label_u = Text("u", font_size=20, color=color_u)
        self.place_at_grid(label_u, 'F4', scale_factor=0.7)
        
        label_v = Text("v", font_size=20, color=color_v)

        # Updaters for dynamic elements
        def update_v(m):
            ang = angle_tracker.get_value()
            new_end = origin_pos + np.array([v_len * np.cos(ang), v_len * np.sin(ang), 0])
            m.put_start_and_end_on(origin_pos, new_end)
        
        def update_v_label(m):
            m.next_to(v_vec.get_end(), UP, buff=0.1)

        def update_area(m):
            p1 = origin_pos
            p2 = u_vec.get_end()
            p3 = v_vec.get_end()
            p4 = p2 + (p3 - p1)
            m.set_points_as_corners([p1, p2, p4, p3, p1])

        v_vec.add_updater(update_v)
        label_v.add_updater(update_v_label)
        area_poly.add_updater(update_area)

        # Perform Animation
        self.play(Create(u_vec), Write(label_u))
        self.play(Create(v_vec), Write(label_v), FadeIn(area_poly))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(highlight_color)

        # Magnitude Formula (Issue 35 Fix)
        formula = Text("|u x v| = |u| |v| sin(theta)", font_size=22)
        self.place_in_area(formula, 'A2', 'A5', scale_factor=0.8)

        # Theta Arc and Label
        theta_arc = Arc(radius=0.5, start_angle=0, angle=angle_tracker.get_value(), 
                        arc_center=origin_pos, color=WHITE)
        theta_label = Text("theta", font_size=18, color=WHITE)
        
        def update_arc(m):
            m.become(Arc(radius=0.5, start_angle=0, angle=angle_tracker.get_value(), 
                         arc_center=origin_pos, color=WHITE))
        
        def update_theta_label(m):
            ang = angle_tracker.get_value() / 2
            pos = origin_pos + np.array([0.8 * np.cos(ang), 0.8 * np.sin(ang), 0])
            m.move_to(pos)

        theta_arc.add_updater(update_arc)
        theta_label.add_updater(update_theta_label)

        self.play(FadeIn(formula))
        self.play(Create(theta_arc), Write(theta_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(highlight_color)

        # Animate the Solar Panel spreading
        self.play(angle_tracker.animate.set_value(85 * DEGREES), run_time=2, rate_func=smooth)
        self.wait(0.5)

        # Animate the Solar Panel closing
        self.play(angle_tracker.animate.set_value(5 * DEGREES), run_time=2, rate_func=smooth)
        self.wait(0.5)

        # Final Rest State
        self.play(angle_tracker.animate.set_value(45 * DEGREES), run_time=1.5)
        self.wait(2)
