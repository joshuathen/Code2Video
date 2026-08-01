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

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(highlight_color)
        
        # Origin for vectors
        origin = self.grid["D4"]
        angle_tracker = ValueTracker(30 * DEGREES)
        
        # Vectors u and v
        u_vec = Vector([2, 0, 0], color=color_u).shift(origin)
        v_vec = Vector([2 * np.cos(30*DEGREES), 2 * np.sin(30*DEGREES), 0], color=color_v).shift(origin)
        
        # Replaced MathTex with Text to avoid LaTeX dependency
        u_label = Text("u", font_size=24, slant=ITALIC, color=color_u).next_to(u_vec.get_end(), RIGHT, buff=0.1)
        v_label = Text("v", font_size=24, slant=ITALIC, color=color_v).next_to(v_vec.get_end(), UP, buff=0.1)

        # Area (Parallelogram)
        area_poly = Polygon(
            origin, 
            u_vec.get_end(), 
            u_vec.get_end() + (v_vec.get_end() - origin), 
            v_vec.get_end(),
            fill_opacity=0.4,
            fill_color=color_area,
            stroke_width=0
        )
        
        # Updaters for movement
        def update_v(m):
            angle = angle_tracker.get_value()
            new_end = origin + np.array([2 * np.cos(angle), 2 * np.sin(angle), 0])
            m.put_start_and_end_on(origin, new_end)

        def update_label_v(m):
            m.next_to(v_vec.get_end(), UP, buff=0.1)

        def update_area(m):
            p1 = origin
            p2 = u_vec.get_end()
            p4 = v_vec.get_end()
            p3 = p2 + (p4 - p1)
            m.set_points_as_corners([p1, p2, p3, p4, p1])

        v_vec.add_updater(update_v)
        v_label.add_updater(update_label_v)
        area_poly.add_updater(update_area)

        self.play(
            GrowArrow(u_vec),
            GrowArrow(v_vec),
            Write(u_label),
            Write(v_label),
            run_time=1
        )
        self.play(FadeIn(area_poly), run_time=1)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(highlight_color)

        # Angle Arc and Label
        theta_arc = Arc(radius=0.5, start_angle=0, angle=angle_tracker.get_value(), arc_center=origin, color=WHITE)
        theta_label = Text("θ", font_size=24, color=WHITE).scale(0.8)
        
        def update_arc(m):
            m.become(Arc(radius=0.5, start_angle=0, angle=angle_tracker.get_value(), arc_center=origin, color=WHITE))
        
        def update_theta_label(m):
            angle = angle_tracker.get_value() / 2
            pos = origin + np.array([0.8 * np.cos(angle), 0.8 * np.sin(angle), 0])
            m.move_to(pos)

        theta_arc.add_updater(update_arc)
        theta_label.add_updater(update_theta_label)

        # Formula (Replaced MathTex with Text)
        formula = Text("|u × v| = |u| |v| sin(θ)", font_size=24, color=WHITE)
        self.place_in_area(formula, "A3", "A6", scale_factor=0.8)

        self.play(Create(theta_arc), Write(theta_label), run_time=1)
        self.play(FadeIn(formula), run_time=1)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(highlight_color)

        # Animate spreading (angle grows to 90 degrees)
        self.play(angle_tracker.animate.set_value(90 * DEGREES), run_time=2, rate_func=linear)
        self.wait(1)

        # Animate closing (angle shrinks to nearly zero)
        self.play(angle_tracker.animate.set_value(0.1 * DEGREES), run_time=2, rate_func=linear)
        self.wait(1)

        # Reset to initial state
        self.play(angle_tracker.animate.set_value(45 * DEGREES), run_time=1)
        self.wait(2)