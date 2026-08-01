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

class Section2Scene(TeachingScene):
    def construct(self):
        title_text = "The Iteration Machine: z² + c"
        lecture_lines = [
            "We focus on the simple rule: z squared plus c.",
            "Squaring magnitude stretches or shrinks the point.",
            "Adding c shifts the point in a specific direction."
        ]
        self.setup_layout(title_text, lecture_lines)

        # Colors
        COLOR_FORMULA = "#00FFFF"
        COLOR_STABLE = "#00FF00"
        COLOR_UNSTABLE = "#FF0000"
        COLOR_SQUARE = YELLOW

        # Assets
        machine_asset_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/machine.svg"

        # === Animation for Lecture Line 1 ===
        # Highlight first line
        self.play(self.lecture[0].animate.set_color(COLOR_FORMULA))

        formula = MathTex("f(z) = z^2 + c", color=COLOR_FORMULA)
        # Fix Issue 21: scale factor 1.0, area A3-B4
        self.place_in_area(formula, 'A3', 'B4', scale_factor=1.0)
        self.play(Write(formula))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(COLOR_SQUARE)
        )
        
        # Setup Plane
        plane = ComplexPlane(
            x_range=[-2.5, 2.5, 1],
            y_range=[-2.5, 2.5, 1],
            x_length=4.5,
            y_length=4.5,
            axis_config={"include_tip": True, "font_size": 18},
            background_line_style={"stroke_opacity": 0.3}
        )
        self.place_in_area(plane, 'C1', 'F6', scale_factor=0.8)
        self.play(Create(plane))

        # Initial point z
        z_val = 0.8 * np.exp(1j * PI / 6) # r=0.8, theta=30 deg
        z_point = Dot(plane.n2p(z_val), color=WHITE)
        z_label = MathTex("z", color=WHITE, font_size=24)
        self.place_at_grid(z_label, 'D3', scale_factor=1.0) # Near z point
        
        self.play(FadeIn(z_point), Write(z_label))
        
        # Machine Asset Integration (Issue 17)
        machine = SVGMobject(machine_asset_path, height=1.0, color=BLUE_B)
        self.place_in_area(machine, 'A5', 'B6', scale_factor=0.8)
        
        self.play(FadeIn(machine))
        self.play(machine.animate.set_color(COLOR_SQUARE), run_time=0.5)
        self.play(machine.animate.set_color(BLUE_B), run_time=0.5)

        # Square it: z^2
        z2_val = z_val**2 # r=0.64, theta=60 deg
        z2_point = Dot(plane.n2p(z2_val), color=COLOR_SQUARE)
        z2_label = MathTex("z^2", color=COLOR_SQUARE, font_size=24)
        self.place_at_grid(z2_label, 'C4', scale_factor=1.0) # Near z^2 point

        # Show magnitude/angle change
        radial_line = Line(plane.coords_to_point(0,0), plane.n2p(z_val), color=WHITE, stroke_width=2)
        radial_line2 = Line(plane.coords_to_point(0,0), plane.n2p(z2_val), color=COLOR_SQUARE, stroke_width=2)

        self.play(Create(radial_line))
        self.wait(0.5)
        
        self.play(
            TransformFromCopy(radial_line, radial_line2),
            TransformFromCopy(z_point, z2_point),
            Write(z2_label),
            run_time=2
        )
        self.play(FadeOut(machine))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(COLOR_STABLE)
        )

        # Translation by c
        c_val = 0.6 - 0.4j
        c_vector = Arrow(
            plane.n2p(z2_val), 
            plane.n2p(z2_val + c_val), 
            buff=0, 
            color=COLOR_FORMULA,
            stroke_width=3,
            max_tip_length_to_length_ratio=0.2
        )
        c_label = MathTex("c", color=COLOR_FORMULA, font_size=24)
        self.place_at_grid(c_label, 'D5', scale_factor=1.0)
        
        final_point = Dot(plane.n2p(z2_val + c_val), color=COLOR_FORMULA)
        
        self.play(Create(c_vector), Write(c_label))
        self.play(FadeIn(final_point))
        self.wait(1)

        # Transition to showing paths
        self.play(
            FadeOut(z_point, z_label, z2_point, z2_label, radial_line, radial_line2, c_vector, c_label, final_point),
        )

        # Show paths: stable spiral and unstable outward
        # Path 1: Stable spiral inward
        c_stable = -0.1 + 0.1j
        z_curr_s = 1.2 + 0.6j
        stable_path_points = []
        for _ in range(15):
            stable_path_points.append(plane.n2p(z_curr_s))
            z_curr_s = z_curr_s**2 + c_stable
            if np.abs(z_curr_s) > 4: break
        
        stable_path = VMobject(color=COLOR_STABLE)
        stable_path.set_points_as_corners(stable_path_points)
        
        stable_label = Text("Stable Path", color=COLOR_STABLE, font_size=24)
        # Fix Issue 22: D2-D3, scale 0.6
        self.place_in_area(stable_label, 'D2', 'D3', scale_factor=0.6)

        # Path 2: Unstable path outward
        c_unstable = 0.35 + 0.1j
        z_curr_u = 1.0 + 0.1j
        unstable_path_points = []
        for _ in range(10):
            unstable_path_points.append(plane.n2p(z_curr_u))
            z_curr_u = z_curr_u**2 + c_unstable
            if np.abs(z_curr_u) > 5: break
            
        unstable_path = VMobject(color=COLOR_UNSTABLE)
        unstable_path.set_points_as_corners(unstable_path_points)
        
        unstable_label = Text("Unstable Path", color=COLOR_UNSTABLE, font_size=24)
        # Fix Issue 23: E5-E6, scale 0.6
        self.place_in_area(unstable_label, 'E5', 'E6', scale_factor=0.6)

        self.play(
            Create(stable_path),
            self.lecture[2].animate.set_color(COLOR_STABLE),
            run_time=2
        )
        self.play(Write(stable_label))
        self.wait(0.5)
        
        self.play(
            Create(unstable_path),
            self.lecture[2].animate.set_color(COLOR_UNSTABLE), 
            run_time=2
        )
        self.play(Write(unstable_label))
        self.wait(2)
