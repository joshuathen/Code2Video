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

class Section3Scene(TeachingScene):
    def construct(self):
        # Colors
        COLOR_U = "#52CEFF"
        COLOR_V = "#C6FF00"
        COLOR_TORQUE = "#FF00FF"
        
        lines = [
            'The result is a vector perpendicular to the plane.',
            'Use the right-hand rule to find the resulting direction.',
            'Swapping u and v flips the direction: it’s non-commutative.'
        ]
        
        self.setup_layout("The Geometry of Direction: The Orthogonality Principle", lines)

        # Assets (Issue 31)
        BOLT_ASSET = "/mmfs1/data/home/jthen/Code2Video/assets/icon/bolt.svg"
        WRENCH_ASSET = "/mmfs1/data/home/jthen/Code2Video/assets/icon/wrench.svg"

        # Base Visual Elements
        # Bolt (Issue 39: Move to C3, scale 0.8)
        bolt = SVGMobject(BOLT_ASSET, color=GRAY_B).set_fill(GRAY_B, opacity=0.8)
        self.place_at_grid(bolt, "C3", scale_factor=0.8)
        
        # Plane representing the surface containing u and v
        plane = Polygon(
            self.grid["B2"] + LEFT*0.2, self.grid["B5"] + RIGHT*0.2, 
            self.grid["D6"] + RIGHT*0.2, self.grid["D3"] + LEFT*0.2,
            color=GRAY_E, fill_opacity=0.2, stroke_width=1
        )
        
        # u is the wrench handle (Issue 31: Use Wrench Asset)
        u_arrow = Arrow(start=self.grid["C3"], end=self.grid["B5"], color=COLOR_U, buff=0)
        wrench = SVGMobject(WRENCH_ASSET, color=COLOR_U)
        wrench.scale(0.4)
        # Calculate angle to align wrench with arrow
        initial_u_vec = self.grid["B5"] - self.grid["C3"]
        wrench_angle = np.arctan2(initial_u_vec[1], initial_u_vec[0])
        wrench.rotate(wrench_angle)
        wrench.move_to(u_arrow.get_center())
        
        u_label = Text("u", color=COLOR_U, slant=ITALIC).scale(0.8)
        u_label.next_to(u_arrow.get_end(), RIGHT, buff=0.1)
        
        # v is the force applied
        v_vec = Arrow(start=self.grid["C3"], end=self.grid["D4"], color=COLOR_V, buff=0)
        v_label = Text("v", color=COLOR_V, slant=ITALIC).scale(0.8)
        v_label.next_to(v_vec.get_end(), DOWN, buff=0.1)
        
        # torque vector (cross product)
        torque_vec = Arrow(start=self.grid["C3"], end=self.grid["A3"], color=COLOR_TORQUE, buff=0)
        torque_label = Text("u × v", color=COLOR_TORQUE).scale(0.8)
        torque_label.next_to(torque_vec.get_end(), UP, buff=0.1)

        # Right-hand rule curl indicator
        curl_arc = ArcBetweenPoints(
            u_arrow.get_start() + (u_arrow.get_vector() * 0.4),
            v_vec.get_start() + (v_vec.get_vector() * 0.4),
            radius=0.5,
            color=WHITE
        ).add_tip(tip_length=0.15)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(COLOR_TORQUE)
        self.play(Create(plane), FadeIn(bolt))
        self.play(GrowArrow(u_arrow), FadeIn(wrench), Write(u_label))
        self.play(GrowArrow(v_vec), Write(v_label))
        self.play(GrowArrow(torque_vec), Write(torque_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(WHITE)
        self.play(Create(curl_arc))
        self.wait(2)
        self.play(FadeOut(curl_arc))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(COLOR_V)
        
        # Prep for swap
        new_u_end = self.grid["D4"]
        new_v_end = self.grid["B5"]
        
        # Non-commutative formula (Issue 38: Use place_in_area)
        nc_text = Text("u × v = -(v × u)", color=WHITE)
        self.place_in_area(nc_text, "A4", "B6", scale_factor=0.6)

        # Reversed torque (Issue 39: ensure not cramped)
        rev_torque_vec = Arrow(start=self.grid["C3"], end=self.grid["E3"], color=COLOR_TORQUE, buff=0)
        rev_torque_label = Text("-(v × u)", color=COLOR_TORQUE).scale(0.8)
        rev_torque_label.next_to(rev_torque_vec.get_end(), DOWN, buff=0.1)
        
        # Target labels for swap
        target_u_label = Text("u", color=COLOR_U, slant=ITALIC).scale(0.8).next_to(new_u_end, DOWN, buff=0.1)
        target_v_label = Text("v", color=COLOR_V, slant=ITALIC).scale(0.8).next_to(new_v_end, RIGHT, buff=0.1)

        # Calculate rotation for wrench
        target_u_vec = new_u_end - self.grid["C3"]
        target_wrench_angle = np.arctan2(target_u_vec[1], target_u_vec[0])
        rotation_diff = target_wrench_angle - wrench_angle

        self.play(
            u_arrow.animate.put_start_and_end_on(self.grid["C3"], new_u_end),
            v_vec.animate.put_start_and_end_on(self.grid["C3"], new_v_end),
            wrench.animate.move_to((self.grid["C3"] + new_u_end)/2).rotate(rotation_diff),
            Transform(u_label, target_u_label),
            Transform(v_label, target_v_label),
            Transform(torque_vec, rev_torque_vec),
            Transform(torque_label, rev_torque_label),
            run_time=2
        )
        self.play(Write(nc_text))
        self.wait(2)
