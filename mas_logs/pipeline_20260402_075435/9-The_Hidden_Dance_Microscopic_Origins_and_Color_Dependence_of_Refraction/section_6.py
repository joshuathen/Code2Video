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

class Section6Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Dispersion causes different colors to focus at different points.",
            "This creates blurry images with unwanted purple fringes.",
            "Engineers use achromatic doublets to correct this color error."
        ]
        self.setup_layout("Application: Chromatic Aberration", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        # Draw a convex lens shape (#ADD8E6)
        lens_1 = Intersection(
            Circle(radius=1.5).shift(LEFT * 1.1),
            Circle(radius=1.5).shift(RIGHT * 1.1),
            color="#ADD8E6", fill_opacity=0.4, stroke_width=2
        )
        self.place_at_grid(lens_1, "C2", scale_factor=0.9)
        
        # Parallel red (#FF0000) and blue (#0000FF) rays enter from left
        ray_offsets = [0.4, 0, -0.4]
        red_rays_in = VGroup(*[
            Line(self.grid["C1"] + UP * y, self.grid["C2"] + UP * y, color="#FF0000", stroke_width=2)
            for y in ray_offsets
        ])
        blue_rays_in = VGroup(*[
            Line(self.grid["C1"] + UP * (y + 0.05), self.grid["C2"] + UP * (y + 0.05), color="#0000FF", stroke_width=2)
            for y in ray_offsets
        ])

        self.play(
            self.lecture[0].animate.set_color(YELLOW),
            Create(lens_1),
            Create(red_rays_in),
            Create(blue_rays_in),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Rays converge on the right; blue focuses at 2 units (C4), red at 3 units (C5)
        focus_blue_pt = self.grid["C4"]
        focus_red_pt = self.grid["C5"]
        
        red_rays_out = VGroup(*[
            Line(self.grid["C2"] + UP * y, focus_red_pt, color="#FF0000", stroke_width=2)
            for y in ray_offsets
        ])
        blue_rays_out = VGroup(*[
            Line(self.grid["C2"] + UP * (y + 0.05), focus_blue_pt, color="#0000FF", stroke_width=2)
            for y in ray_offsets
        ])
        
        # Label "Aberration" (#FF0000) - Fixed Position for Issue 39
        aberration_label = Text("Aberration", font_size=24, color="#FF0000")
        self.place_at_grid(aberration_label, "B4", scale_factor=0.8)
        
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(PURPLE),
            Create(red_rays_out),
            Create(blue_rays_out),
            Write(aberration_label),
            run_time=2
        )
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # Add a second lens shape (#B0C4DE) to the right.
        # Concave lens shape for the doublet
        rect_shape = Rectangle(width=0.3, height=1.5)
        c_subtract = Circle(radius=1.2)
        lens_2 = Difference(
            rect_shape, 
            Union(c_subtract.copy().shift(LEFT*1.3), c_subtract.copy().shift(RIGHT*1.3)),
            color="#B0C4DE", fill_opacity=0.4, stroke_width=2
        )
        self.place_at_grid(lens_2, "C3", scale_factor=0.9)
        
        # Rays now converge at a single point (#00FF00) at C6
        focus_corrected = self.grid["C6"]
        
        # Intermediate paths from lens 1 to lens 2
        red_mid = VGroup(*[
            Line(self.grid["C2"] + UP * y, self.grid["C3"] + UP * (y * 0.7), color="#FF0000", stroke_width=2)
            for y in ray_offsets
        ])
        blue_mid = VGroup(*[
            Line(self.grid["C2"] + UP * (y + 0.05), self.grid["C3"] + UP * ((y + 0.05) * 0.7), color="#0000FF", stroke_width=2)
            for y in ray_offsets
        ])
        
        # Final converged paths from lens 2 to focus point
        # Rays change color to green at the meeting point as per instructions
        red_out_corrected = VGroup(*[
            Line(self.grid["C3"] + UP * (y * 0.7), focus_corrected, color="#00FF00", stroke_width=2)
            for y in ray_offsets
        ])
        blue_out_corrected = VGroup(*[
            Line(self.grid["C3"] + UP * ((y + 0.05) * 0.7), focus_corrected, color="#00FF00", stroke_width=2)
            for y in ray_offsets
        ])

        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(GREEN),
            FadeOut(aberration_label), # Addressing Issue 40: ensure label fades out
            FadeOut(red_rays_out),
            FadeOut(blue_rays_out),
            Create(lens_2),
            Create(red_mid),
            Create(blue_mid),
            Create(red_out_corrected),
            Create(blue_out_corrected),
            run_time=2
        )
        self.wait(3)
