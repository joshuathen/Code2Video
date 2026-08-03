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

class Section7Scene(TeachingScene):
    def construct(self):
        title = "Summary & Real-World Connection"
        lines = [
            "Dandelin's spheres bridge 3D shapes and 2D curves.",
            "This geometry explains orbits, reflectors, and satellite dishes.",
            "Pure geometry reveals the deep soul of conic sections."
        ]
        self.setup_layout(title, lines)

        # Colors
        COLOR_CONE = "#87CEEB" # Sky Blue
        COLOR_SPHERE = "#FFA07A" # Light Salmon
        COLOR_ELLIPSE = "#F0E68C" # Khaki
        COLOR_RAY = "#FFFF00" # Yellow
        COLOR_SUMMARY = "#00FA9A" # Medium Spring Green

        # === Animation for Lecture Line 1 ===
        # Dandelin's spheres bridge 3D shapes and 2D curves.
        self.play(self.lecture[0].animate.set_color(COLOR_CONE))

        # 3D Representation (Simplified 2D Projection)
        cone_outline = Triangle(color=COLOR_CONE).scale(1.2)
        sphere1 = Circle(radius=0.4, color=COLOR_SPHERE, fill_opacity=0.3).move_to(cone_outline.get_center() + UP*0.2)
        sphere2 = Circle(radius=0.2, color=COLOR_SPHERE, fill_opacity=0.3).move_to(cone_outline.get_top() + DOWN*0.4)
        three_d_model = VGroup(cone_outline, sphere1, sphere2)
        
        # Fix: Issue 45 - Repositioning three_d_model to avoid crowding
        self.place_in_area(three_d_model, 'B2', 'D3')

        # 2D Representation
        ellipse = Ellipse(width=1.5, height=1.0, color=COLOR_ELLIPSE)
        foci = VGroup(
            Dot(ellipse.get_center() + LEFT*0.4, color=COLOR_ELLIPSE, radius=0.04),
            Dot(ellipse.get_center() + RIGHT*0.4, color=COLOR_ELLIPSE, radius=0.04)
        )
        two_d_model = VGroup(ellipse, foci)
        self.place_in_area(two_d_model, "B4", "C6")

        bridge_arrow = Arrow(three_d_model.get_right(), two_d_model.get_left(), color=WHITE)

        self.play(Create(three_d_model))
        self.wait(0.5)
        self.play(GrowArrow(bridge_arrow))
        self.play(Create(two_d_model))
        self.wait(2)

        # === Animation for Lecture Line 2 ===
        # This geometry explains orbits, reflectors, and satellite dishes.
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(COLOR_RAY),
            FadeOut(three_d_model),
            FadeOut(two_d_model),
            FadeOut(bridge_arrow)
        )

        # Satellite Dish Asset: Issue 28
        # Using SVGMobject for the satellite dish as requested.
        satellite_dish = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/satellite.svg", color=WHITE)
        self.place_in_area(satellite_dish, "E2", "F5", scale_factor=1.2)
        
        # Focal point of the parabola
        focus_point = self.grid["D4"]
        focus_dot = Dot(focus_point, color=COLOR_RAY)
        focus_label = Text("Focus", font_size=16, color=COLOR_RAY)
        
        # Fix: Issue 46 - Corrected focus label position to D4 with scale 0.8
        self.place_at_grid(focus_label, 'D4', scale_factor=0.8)
        # Offset label slightly from dot for clarity
        focus_label.next_to(focus_dot, UP, buff=0.2)

        # Incoming rays - Parallel rays reflecting to focus
        rays = VGroup()
        ray_x_offsets = [-1.5, -0.75, 0, 0.75, 1.5]
        for dx in ray_x_offsets:
            # Start above the screen
            start_pos = np.array([focus_point[0] + dx, 2.5, 0])
            # Hit the dish (roughly at row E/F height)
            hit_pos = np.array([start_pos[0], self.grid["E4"][1] - 0.4, 0])
            in_ray = Line(start_pos, hit_pos, color=COLOR_RAY, stroke_width=2)
            out_ray = Line(hit_pos, focus_point, color=COLOR_RAY, stroke_width=2)
            rays.add(VGroup(in_ray, out_ray))

        self.play(DrawBorderThenFill(satellite_dish))
        self.play(FadeIn(focus_dot), Write(focus_label))
        
        # Animate rays reflecting
        self.play(
            *[Create(r[0]) for r in rays],
            run_time=1
        )
        self.play(
            *[Create(r[1]) for r in rays],
            run_time=1
        )
        
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # Pure geometry reveals the deep soul of conic sections.
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(COLOR_SUMMARY)
        )

        summary_text = Text("Geometry and Algebra Unified", font_size=24, color=COLOR_SUMMARY)
        # Fix: Issue 44 - Repositioned summary_text to 'A4'-'B6' area to avoid overlap
        self.place_in_area(summary_text, 'A4', 'B6', scale_factor=0.8)

        self.play(Write(summary_text))
        self.play(Indicate(summary_text, color=COLOR_SUMMARY))
        
        self.wait(3)
