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
        # Initializing scene layout
        self.setup_layout(
            "Entering the Complex Plane: Analytic Continuation",
            [
                "We now extend s into the complex plane.",
                "Analytic continuation stretches the function's reach.",
                "A warped grid reveals the function's hidden landscape."
            ]
        )

        # Define specific hex colors for synchronization
        color_plane = YELLOW_A
        color_shading = "#00AEFF"
        color_warp = GREEN_A

        # Define grid bounds for the complex plane background
        plane_center = (self.grid['A1'] + self.grid['F6']) / 2
        plane_width = self.grid['F6'][0] - self.grid['A1'][0]
        plane_height = self.grid['A1'][1] - self.grid['F6'][1]

        # Create Complex Plane
        plane = ComplexPlane(
            x_range=[-2, 5, 1],
            y_range=[-3, 3, 1],
            x_length=plane_width + 1,
            y_length=plane_height + 1,
            background_line_style={"stroke_opacity": 0.5, "stroke_width": 1}
        ).move_to(plane_center)
        
        labels = plane.get_axis_labels(x_label=Text("Re"), y_label=Text("Im"))

        # === Animation for Lecture Line 1 ===
        # Connection: Plane and labels match yellow-themed lecture line
        self.play(self.lecture[0].animate.set_color(color_plane))
        
        # Identity formula positioned to avoid overlap (Issue 41)
        formula = Text("s = σ + it", font_size=32, color=color_plane)
        self.place_in_area(formula, 'B1', 'B6', scale_factor=0.8)
        
        self.play(
            Create(plane),
            Write(labels),
            Write(formula),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Connection: Shading and labels match the semi-transparent blue #00AEFF
        self.play(self.lecture[1].animate.set_color(color_shading))
        
        # Calculate region for Re(s) > 1
        p_start = plane.coords_to_point(1, 3.5)
        p_end = plane.coords_to_point(5.5, -3.5)
        shading = Rectangle(
            width=p_end[0] - p_start[0],
            height=p_start[1] - p_end[1],
            fill_color=color_shading,
            fill_opacity=0.3,
            stroke_width=0
        ).move_to((p_start + p_end) / 2)

        # Labels for the domain (Issue 42 & 43)
        domain_label = Text("Convergence Region", font_size=24, color=color_shading)
        self.place_at_grid(domain_label, 'C3', scale_factor=1.1)
        
        continuation_label = Text("Analytic Continuation", font_size=24, color=WHITE)
        self.place_in_area(continuation_label, 'D2', 'D4', scale_factor=0.7)

        self.play(
            FadeIn(shading),
            Write(domain_label),
            Write(continuation_label),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Connection: Warping effect synchronized with green-themed lecture line
        self.play(self.lecture[2].animate.set_color(color_warp))

        # Nonlinear warping function to simulate "rubber sheet" metaphor
        def rubber_sheet_warp(p):
            x, y, z = p
            # Distort the plane more on the left side (where analytic continuation happens)
            factor = np.exp(-0.2 * x) 
            new_x = x + 0.3 * factor * np.sin(y * 1.5)
            new_y = y + 0.2 * factor * np.cos(x * 1.5)
            return np.array([new_x, new_y, z])

        # Group elements to warp together
        warping_group = VGroup(plane, shading, domain_label, continuation_label)

        self.play(
            warping_group.animate.apply_function(rubber_sheet_warp),
            formula.animate.set_color(color_warp),
            run_time=4,
            rate_func=slow_into
        )
        
        self.wait(3)
