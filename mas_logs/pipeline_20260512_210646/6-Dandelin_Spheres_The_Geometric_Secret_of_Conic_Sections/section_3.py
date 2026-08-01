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
        # Setup the scene layout
        self.setup_layout(
            "The Dandelin Construction", 
            [
                "Place two spheres inside the cone.", 
                "Each sphere touches the cone along a circle.", 
                "They also touch the cutting plane at F1 and F2.", 
                "These \"kissing points\" on the plane are the foci.", 
                "Let's see why this construction works."
            ]
        )

        # Define colors for elements matching lecture lines
        COLOR_SPHERES = RED_A
        COLOR_CONTACT = YELLOW
        COLOR_FOCI = PINK
        COLOR_PLANE = BLUE_C
        COLOR_CONE = LIGHT_GRAY

        # Define Geometry relative to a local origin
        # Cone: Apex at (0, -2), sides up to (2, 2)
        cone_lines = VGroup(
            Line(start=[0, -2, 0], end=[2, 2, 0], color=COLOR_CONE),
            Line(start=[0, -2, 0], end=[-2, 2, 0], color=COLOR_CONE)
        )
        
        # Cutting Plane: Slanted line
        plane_line = Line(start=[-1.8, 1.2, 0], end=[1.8, -0.8, 0], color=COLOR_PLANE, stroke_width=4)
        
        # Spheres as circles in cross-section
        # Sphere 1 (Top): Larger
        sphere_top = Circle(radius=0.9, color=COLOR_SPHERES, fill_opacity=0.2, fill_color=COLOR_SPHERES).move_to([0, 0.8, 0])
        # Sphere 2 (Bottom): Smaller, closer to apex
        sphere_bottom = Circle(radius=0.4, color=COLOR_SPHERES, fill_opacity=0.2, fill_color=COLOR_SPHERES).move_to([0, -1.2, 0])
        
        # Focal points (Visual approximation of tangency)
        # Positioned relative to spheres before group scaling
        f1_dot = Dot(point=sphere_top.get_center() + np.array([0.5, -0.7, 0]), color=COLOR_FOCI)
        f2_dot = Dot(point=sphere_bottom.get_center() + np.array([-0.25, 0.3, 0]), color=COLOR_FOCI)
        
        # Labels for tangency points on the plane
        # Offset slightly for better readability when scaled (Issue 36)
        f1_label = Text("F1", font_size=18, color=COLOR_FOCI).next_to(f1_dot, UR, buff=0.1)
        f2_label = Text("F2", font_size=18, color=COLOR_FOCI).next_to(f2_dot, DL, buff=0.1)

        # Contact horizontal lines (circles of tangency in cross-section)
        contact_line_top = Line(
            start=sphere_top.get_center() + LEFT*0.9, 
            end=sphere_top.get_center() + RIGHT*0.9, 
            color=COLOR_CONTACT, stroke_width=3
        )
        contact_line_bottom = Line(
            start=sphere_bottom.get_center() + LEFT*0.4, 
            end=sphere_bottom.get_center() + RIGHT*0.4, 
            color=COLOR_CONTACT, stroke_width=3
        )

        # Combine all geometry to ensure they scale and move together
        viz_group = VGroup(
            cone_lines, plane_line, 
            sphere_top, sphere_bottom, 
            f1_dot, f2_dot, f1_label, f2_label,
            contact_line_top, contact_line_bottom
        )
        
        # Place visualization in the larger grid area A1-F6 at scale 0.8 (Issue 35)
        self.place_in_area(viz_group, "A1", "F6", scale_factor=0.8)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(COLOR_SPHERES)
        self.play(Create(cone_lines), Create(plane_line))
        self.play(FadeIn(sphere_top), FadeIn(sphere_bottom))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(COLOR_CONTACT)
        self.play(Create(contact_line_top), Create(contact_line_bottom))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(COLOR_FOCI)
        self.play(Create(f1_dot), Create(f2_dot))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(COLOR_FOCI)
        self.play(Write(f1_label), Write(f2_label))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(WHITE)
        self.play(
            Indicate(sphere_top, color=COLOR_SPHERES),
            Indicate(sphere_bottom, color=COLOR_SPHERES),
            Indicate(plane_line, color=COLOR_PLANE),
            run_time=2
        )
        self.wait(2)
