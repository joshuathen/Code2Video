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
        # Setup layout with specific title and lines
        lines = [
            "This boundary is the Julia set, a complex fractal.",
            "It is the thin, chaotic edge between staying and escaping.",
            "Zooming in reveals self-similar patterns repeating at every scale.",
            "Small changes here lead to radically different future outcomes.",
            "The Julia set's structure depends entirely on the constant c."
        ]
        self.setup_layout("The Julia Set: The Edge of Chaos", lines)

        # Helper to generate Julia Set boundary points via backward iteration
        def get_julia_points(c_val, num_pts=800):
            pts = []
            z = (1 + (1 - 4 * c_val)**0.5) / 2 # Fixed point
            for _ in range(num_pts + 100):
                if np.random.random() > 0.5:
                    z = (z - c_val)**0.5
                else:
                    z = -(z - c_val)**0.5
                if _ >= 100:
                    pts.append([z.real, z.imag, 0])
            return pts

        # Initial constant
        c_val = -0.8 + 0.156j
        julia_color = "#FF8C00"

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(julia_color))
        
        # Create Julia set as a group of small dots
        julia_points = get_julia_points(c_val)
        julia_dots = VGroup(*[Dot(point=p, radius=0.015, color=julia_color) for p in julia_points])
        
        # Container to hold the fractal for easier transformation
        fractal_container = VGroup(julia_dots)
        # Issue 35: Reduce scale factor to avoid overlapping lecture text
        self.place_in_area(fractal_container, "A1", "F6", scale_factor=1.2)
        
        self.play(Create(fractal_container), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(WHITE))
        
        # Highlight "boundary" concept: Pulse the dots
        self.play(
            julia_dots.animate.scale(1.1),
            rate_func=there_and_back,
            run_time=1.5
        )
        self.wait(0.5)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(WHITE))
        
        # Perform deep zoom
        # Choose a sub-region to zoom into (top right branch)
        zoom_target = self.grid["B5"]
        zoom_factor = 6
        
        # Prepare self-similarity display
        self.play(
            fractal_container.animate.scale(zoom_factor, about_point=zoom_target),
            run_time=3
        )
        
        # Flash a sub-region to show self-similarity
        sub_region_circle = Circle(radius=0.3, color=WHITE).move_to(self.grid["C4"])
        self.play(Create(sub_region_circle))
        self.play(Flash(sub_region_circle, color=WHITE))
        self.play(FadeOut(sub_region_circle))
        
        # Restore view for next part
        self.play(fractal_container.animate.scale(1/zoom_factor, about_point=zoom_target), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Use a distinguishable color for the points
        point_color = RED
        self.play(self.lecture[3].animate.set_color(point_color))
        
        # Find a point near the Julia set boundary
        z0 = complex(0.1, 0.1)
        z1 = z0 + 0.0001 # Extremely close
        
        dot0 = Dot(point=[z0.real, z0.imag, 0], color=point_color, radius=0.05)
        dot1 = Dot(point=[z1.real, z1.imag, 0], color=YELLOW, radius=0.05)
        
        # Add to container
        fractal_container.add(dot0, dot1)
        
        # Label for the points
        # Issue 36: Position label at F3 with scale factor 0.8
        label_div = Text("Chaotic divergence", font_size=16, color=WHITE)
        self.place_at_grid(label_div, "F3", scale_factor=0.8)
        self.add(label_div)

        # Iteration paths
        path0 = [dot0.get_center()]
        path1 = [dot1.get_center()]
        
        curr_z0 = z0
        curr_z1 = z1
        
        # Manually iterate a few steps to show divergence
        # Corrected scale multiplier to 1.2 to match fractal_container's scale
        for _ in range(5):
            curr_z0 = curr_z0**2 + c_val
            curr_z1 = curr_z1**2 + c_val
            path0.append(np.array([curr_z0.real, curr_z0.imag, 0]) * 1.2 + fractal_container.get_center())
            path1.append(np.array([curr_z1.real, curr_z1.imag, 0]) * 1.2 + fractal_container.get_center())

        # Animate the divergence
        trace0 = TracedPath(dot0.get_center, stroke_color=point_color, stroke_width=2)
        trace1 = TracedPath(dot1.get_center, stroke_color=YELLOW, stroke_width=2)
        self.add(trace0, trace1)

        # Manual animation of steps
        for i in range(1, len(path0)):
            self.play(
                dot0.animate.move_to(path0[i]),
                dot1.animate.move_to(path1[i]),
                run_time=0.5
            )
        
        self.wait(1)
        self.play(FadeOut(dot0), FadeOut(dot1), FadeOut(trace0), FadeOut(trace1), FadeOut(label_div))

        # === Animation for Lecture Line 5 ===
        morph_color = "#00BFFF"
        self.play(self.lecture[4].animate.set_color(morph_color))
        
        # Varying c to morph shape
        c_target = complex(0.285, 0.01)
        
        new_julia_points = get_julia_points(c_target)
        new_julia_dots = VGroup(*[Dot(point=p, radius=0.015, color=morph_color) for p in new_julia_points])
        # Issue 37: Reduce scale factor to 1.2 to stay within boundaries
        self.place_in_area(new_julia_dots, "A1", "F6", scale_factor=1.2)
        
        self.play(
            Transform(julia_dots, new_julia_dots),
            run_time=3,
            rate_func=smooth
        )
        
        self.wait(2)
