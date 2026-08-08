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
        # Setup layout
        title = "The Magic Trick: Inserting the Spheres"
        lecture_lines = [
            "- Nest two spheres between the cone and plane.",
            "- Plane contact points will become our two foci.",
            "- We call these the Dandelin Spheres."
        ]
        self.setup_layout(title, lecture_lines)

        # Define colors
        CONE_COLOR = "#888888"
        PLANE_COLOR = "#FFFFFF"
        SPHERE_COLOR = "#FFD700"
        FOCI_COLOR = "#FF0000"

        # 3D Construction Group
        three_d_view = VGroup()

        # 1. Cone (Visual representation)
        # Using a low-opacity cone to see through it
        cone = Cone(base_radius=1.5, height=4.0, direction=UP, 
                   fill_opacity=0.1, color=CONE_COLOR, stroke_width=1)
        cone.rotate(PI, axis=RIGHT)
        three_d_view.add(cone)

        # 2. Slicing Plane
        plane = Rectangle(width=4, height=4, fill_opacity=0.2, 
                          fill_color=PLANE_COLOR, color=PLANE_COLOR)
        plane.rotate(60 * DEGREES, axis=RIGHT)
        plane.rotate(10 * DEGREES, axis=UP)
        plane.move_to(UP * 0.2)
        three_d_view.add(plane)

        # Positioning fix from Issue 30: Moved further right to A3-F6 and reduced scale
        self.place_in_area(three_d_view, 'A3', 'F6', scale_factor=0.7)

        # Pre-calculate positions relative to the 3D view
        view_center = three_d_view.get_center()

        # Small Sphere (above plane)
        small_sphere = Sphere(radius=0.4, fill_opacity=0.6, color=SPHERE_COLOR)
        small_sphere.move_to(view_center + UP * 1.0 + RIGHT * 0.1)
        
        # Large Sphere (below plane)
        large_sphere = Sphere(radius=0.9, fill_opacity=0.6, color=SPHERE_COLOR)
        large_sphere.move_to(view_center + DOWN * 0.8 + LEFT * 0.1)

        # Contact Points (Foci)
        f1_pos = small_sphere.get_center() + DOWN * 0.4 + LEFT * 0.05
        f1 = Dot(point=f1_pos, color=FOCI_COLOR, radius=0.08)
        f1_label = MathTex("F_1", color=FOCI_COLOR, font_size=24)
        f1_label.next_to(f1, UR, buff=0.1)

        f2_pos = large_sphere.get_center() + UP * 0.9 + RIGHT * 0.05
        f2 = Dot(point=f2_pos, color=FOCI_COLOR, radius=0.08)
        f2_label = MathTex("F_2", color=FOCI_COLOR, font_size=24)
        f2_label.next_to(f2, DL, buff=0.1)

        # Label for the spheres
        dandelin_label = Text("Dandelin Spheres", font_size=18, color=SPHERE_COLOR)
        # Positioning fix from Issue 31: Centered horizontally in the visual area
        self.place_in_area(dandelin_label, 'F4', 'F5', scale_factor=0.9)

        # === Animation for Lecture Line 1 ===
        # Highlight lecture line and reveal cone/plane
        self.play(self.lecture[0].animate.set_color(SPHERE_COLOR))
        self.play(Create(cone), Create(plane))
        self.wait(1)

        # Animate spheres entering the cone
        self.play(
            FadeIn(small_sphere, shift=DOWN),
            FadeIn(large_sphere, shift=UP)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Transition lecture highlight to foci focus
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(FOCI_COLOR)
        )

        # Mark the contact points as F1 and F2
        self.play(
            Create(f1), Write(f1_label),
            Create(f2), Write(f2_label)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Final highlight of the Dandelin Spheres
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(SPHERE_COLOR)
        )
        
        # Add visual emphasis to the spheres and show the collective name
        self.play(
            small_sphere.animate.set_stroke(SPHERE_COLOR, width=2),
            large_sphere.animate.set_stroke(SPHERE_COLOR, width=2),
            Write(dandelin_label)
        )
        self.wait(2)
