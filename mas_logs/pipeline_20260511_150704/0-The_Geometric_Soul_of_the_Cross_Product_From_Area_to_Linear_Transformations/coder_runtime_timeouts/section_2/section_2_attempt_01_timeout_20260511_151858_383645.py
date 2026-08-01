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
        # Setup the layout with the specific title and lecture lines
        self.setup_layout(
            "The Geometry of Magnitude: Parallelograms", 
            [
                "In 3D, the cross product's magnitude represents area.", 
                "This area equals magnitude u times v times sine theta.", 
                "As the angle grows, the solar panel's area increases."
            ]
        )
        
        # Colors
        color_u = "#52CEFF"
        color_v = "#C6FF00"
        color_area = "#FFD700"
        highlight_color = YELLOW

        # Origin point for the right side
        origin = self.grid["D2"]
        angle_tracker = ValueTracker(30 * DEGREES)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(highlight_color)

        # Vectors u and v
        u_vec = Vector([2.2, 0, 0], color=color_u).shift(origin)
        v_vec = Vector([1.8, 0, 0], color=color_v).shift(origin)
        
        # Vector labels (Text instead of MathTex for safety)
        u_label = Text("u", font_size=24, color=color_u, slant=ITALIC)
        v_label = Text("v", font_size=24, color=color_v, slant=ITALIC)

        # Parallelogram Area
        area_poly = Polygon(
            origin, 
            origin + RIGHT, 
            origin + RIGHT + UP, 
            origin + UP,
            fill_opacity=0.4,
            fill_color=color_area,
            stroke_width=1,
            stroke_color=color_area
        )

        def update_v(m):
            angle = angle_tracker.get_value()
            new_end = origin + np.array([2.0 * np.cos(angle), 2.0 * np.sin(angle), 0])
            m.put_start_and_end_on(origin, new_end)

        def update_v_label(m):
            m.next_to(v_vec.get_end(), UP, buff=0.1)

        def update_area(m):
            p1 = origin
            p2 = u_vec.get_end()
            p4 = v_vec.get_end()
            p3 = p2 + (p4 - p1)
            m.set_points_as_corners([p1, p2, p3, p4, p1])

        # Initialization
        v_vec.add_updater(update_v)
        v_label.add_updater(update_v_label)
        area_poly.add_updater(update_area)
        
        u_label.next_to(u_vec.get_end(), DOWN, buff=0.1)

        self.play(GrowArrow(u_vec), Write(u_label), run_time=1)
        self.play(GrowArrow(v_vec), Write(v_label), run_time=1)
        self.play(FadeIn(area_poly), run_time=1)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(highlight_color)

        # Theta Arc and Label
        theta_arc = Arc(radius=0.5, start_angle=0, angle=angle_tracker.get_value(), arc_center=origin, color=WHITE)
        theta_text = Text("θ", font_size=24, color=WHITE)

        def update_arc(m):
            m.become(Arc(radius=0.5, start_angle=0, angle=angle_tracker.get_value(), arc_center=origin, color=WHITE))
        
        def update_theta_pos(m):
            angle = angle_tracker.get_value() / 2
            pos = origin + np.array([0.7 * np.cos(angle), 0.7 * np.sin(angle), 0])
            m.move_to(pos)

        theta_arc.add_updater(update_arc)
        theta_text.add_updater(update_theta_pos)

        # Magnitude Formula
        formula = Text("|u x v| = |u| |v| sin(θ)", font_size=24, color=WHITE)
        self.place_in_area(formula, "A2", "A6", scale_factor=0.8)

        self.play(Create(theta_arc), FadeIn(theta_text), run_time=1)
        self.play(Write(formula), run_time=1)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(highlight_color)

        # Spreading (angle grows)
        self.play(angle_tracker.animate.set_value(90 * DEGREES), run_time=2)
        self.wait(1)

        # Closing (angle shrinks)
        self.play(angle_tracker.animate.set_value(5 * DEGREES), run_time=2)
        self.wait(1)

        # Return to pleasant angle
        self.play(angle_tracker.animate.set_value(45 * DEGREES), run_time=1)
        self.wait(2)
