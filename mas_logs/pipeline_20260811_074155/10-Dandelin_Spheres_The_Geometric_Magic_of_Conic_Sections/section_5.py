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
        # 1. Setup layout
        lines = [
            "Changing the plane's angle creates parabolas and hyperbolas.",
            "Dandelin spheres reveal the logic behind all conic sections.",
            "Geometry bridges the gap between 3D space and 2D curves."
        ]
        self.setup_layout("Generalization and Real-World Echoes", lines)
        
        # Colors based on storyboard requirements
        PLANE_COLOR = "#1E90FF"
        SPHERE_COLOR = "#D3D3D3"
        CONIC_COLOR = "#FFFF00"
        CONE_COLOR = "#666666"

        # Tracker for the angle of the slicing plane
        angle_tracker = ValueTracker(0.4)
        
        # Center for the 3D-to-2D diagram representation
        diag_center = self.grid["D4"]
        
        # Asset: Airplane icon used as the slicing plane handle
        airplane = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/airplane.svg", color=PLANE_COLOR)
        airplane.scale(0.3)
        airplane_orig_points = airplane.points.copy()
        
        # Slicing plane pivot
        pivot = diag_center + DOWN * 0.5
        
        def update_airplane(m):
            m.points = airplane_orig_points.copy()
            m.rotate(angle_tracker.get_value())
            m.move_to(pivot + rotate_vector(RIGHT * 1.5, angle_tracker.get_value()))
        airplane.add_updater(update_airplane)

        # Cone (2D cross section)
        apex = diag_center + UP * 1.5
        wall_left = Line(apex, diag_center + LEFT * 2 + DOWN * 1.5, color=CONE_COLOR)
        wall_right = Line(apex, diag_center + RIGHT * 2 + DOWN * 1.5, color=CONE_COLOR)
        cone = VGroup(wall_left, wall_right)
        
        # Plane line
        plane_line = Line(LEFT * 1.8, RIGHT * 1.8, color=PLANE_COLOR, stroke_width=4)
        def update_plane(m):
            m.set_angle(angle_tracker.get_value())
            m.move_to(pivot)
        plane_line.add_updater(update_plane)
        
        # Dynamic Conic Type Labels (optimized: pre-created and toggled by updater)
        lbl_ellipse = Text("ELLIPSE", font_size=24, color=CONIC_COLOR)
        lbl_parabola = Text("PARABOLA", font_size=24, color=CONIC_COLOR)
        lbl_hyperbola = Text("HYPERBOLA", font_size=24, color=CONIC_COLOR)
        labels = VGroup(lbl_ellipse, lbl_parabola, lbl_hyperbola)
        for l in labels:
            self.place_at_grid(l, "A4", scale_factor=1.0)
            l.set_opacity(0)
            
        def update_labels(m):
            theta = angle_tracker.get_value()
            if abs(theta) < 0.75: idx = 0
            elif abs(theta) < 0.82: idx = 1
            else: idx = 2
            for i, l in enumerate(m):
                l.set_opacity(1 if i == idx else 0)
        labels.add_updater(update_labels)

        # Dandelin Spheres (2D side view)
        sphere1 = Circle(color=SPHERE_COLOR, stroke_width=2).set_opacity(0.6)
        sphere2 = Circle(color=SPHERE_COLOR, stroke_width=2).set_opacity(0.6)
        spheres = VGroup(sphere1, sphere2)
        
        def update_spheres(m):
            theta = angle_tracker.get_value()
            H = 1.5 # apex y relative to center
            Pb = -0.5 # pivot y relative to center
            k1 = np.sqrt(2) # sqrt(1 + m_cone^2)
            k2 = 1.0 / np.cos(theta) # sqrt(1 + tan^2)
            b = Pb
            
            # Sphere 1 center y (relative to diag_center)
            y1 = (k2 * H + k1 * b) / (k1 + k2)
            r1 = (H - y1) / k1
            m[0].set_width(max(0.01, 2*r1)).move_to(diag_center + UP * y1)
            
            # Sphere 2 center y
            if abs(k1 - k2) > 0.05:
                y2 = (k1 * b - k2 * H) / (k1 - k2)
                r2 = abs(H - y2) / k1
                m[1].set_width(max(0.01, 2*r2)).move_to(diag_center + UP * y2)
                m[1].set_opacity(0.6 if -2.5 < y2 < 2.5 else 0)
            else:
                m[1].set_opacity(0)
            
            m[0].set_opacity(0.6 if -2.5 < y1 < 2.5 else 0)

        spheres.add_updater(update_spheres)

        # Resulting Conic Curves
        ellipse_c = Ellipse(width=1.2, height=0.8, color=CONIC_COLOR).set_stroke(width=4)
        parabola_c = FunctionGraph(lambda x: 0.5*x**2 - 0.4, x_range=[-1.0, 1.0], color=CONIC_COLOR).set_stroke(width=4)
        hyperbola_c = VGroup(
            FunctionGraph(lambda x: np.sqrt(x**2 + 0.2) - 0.4, x_range=[-0.8, 0.8], color=CONIC_COLOR),
            FunctionGraph(lambda x: -np.sqrt(x**2 + 0.2) + 0.4, x_range=[-0.8, 0.8], color=CONIC_COLOR)
        ).set_stroke(width=4)

        # Positions per issues 22, 23, 24 for side-by-side comparison
        self.place_at_grid(ellipse_c, 'B4', scale_factor=0.7)
        self.place_at_grid(parabola_c, 'D4', scale_factor=0.7)
        self.place_at_grid(hyperbola_c, 'F4', scale_factor=0.7)

        # === Animation for Lecture Line 1 ===
        # "Changing the plane's angle creates parabolas and hyperbolas."
        self.play(self.lecture[0].animate.set_color(PLANE_COLOR))
        self.play(Create(cone))
        self.play(FadeIn(airplane), Create(plane_line), FadeIn(labels))
        self.wait(0.5)
        # Shift through conic types
        self.play(angle_tracker.animate.set_value(0.785), run_time=2) # Transition to Parabola
        self.wait(1)
        self.play(angle_tracker.animate.set_value(1.1), run_time=2) # Transition to Hyperbola
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "Dandelin spheres reveal the logic behind all conic sections."
        self.play(self.lecture[1].animate.set_color(SPHERE_COLOR))
        self.play(angle_tracker.animate.set_value(0.4), run_time=1.5)
        self.play(FadeIn(spheres))
        self.wait(0.5)
        # Animate spheres resizing as plane tilts
        self.play(angle_tracker.animate.set_value(1.15), run_time=3.5)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "Geometry bridges the gap between 3D space and 2D curves."
        self.play(self.lecture[2].animate.set_color(CONIC_COLOR))
        self.play(
            FadeOut(cone), FadeOut(plane_line), FadeOut(airplane), 
            FadeOut(labels), FadeOut(spheres)
        )
        # Show all generalized conic curves
        self.play(
            FadeIn(ellipse_c),
            FadeIn(parabola_c),
            FadeIn(hyperbola_c)
        )
        self.wait(2)
