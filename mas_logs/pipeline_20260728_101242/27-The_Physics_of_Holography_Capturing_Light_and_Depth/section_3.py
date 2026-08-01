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

class Section3Scene(TeachingScene):
    def construct(self):
        # Setup the layout with section title and lecture lines
        title_text = "The Mechanism: Diffraction as a Decoder"
        lecture_lines = [
            "- Diffraction is the bending of light around small obstacles.",
            "- A hologram acts as a complex microscopic diffraction grating.",
            "- This grating reconstructs the original wavefront when illuminated."
        ]
        self.setup_layout(title_text, lecture_lines)

        # Colors defined for consistency across animation stages
        COLOR_GRATING = "#CCCCCC"
        COLOR_DIFFRACTION = "#00FFFF"
        COLOR_DOTS = "#FFFF00"

        # === Animation for Lecture Line 1 ===
        # "Diffraction is the bending of light around small obstacles."
        self.lecture[0].set_color(COLOR_GRATING)
        
        # Grating (comb structure) - Using Asset
        # Issue 31 Fix: Positioned in area B3-E3 to avoid crowding lecture notes
        comb = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/comb.svg", color=COLOR_GRATING)
        self.place_in_area(comb, 'B3', 'E3', scale_factor=0.8)
        
        # Incoming parallel rays hitting the comb from the left
        rays_in = VGroup(*[
            Line(LEFT * 1.5, ORIGIN, color=COLOR_GRATING, stroke_width=2)
            for _ in range(5)
        ])
        comb_center = comb.get_center()
        comb_height = comb.height
        for i, r in enumerate(rays_in):
            # Distribute rays vertically to cover the comb
            y_offset = (i - 2) * (comb_height / 4)
            r.move_to(comb_center + LEFT * 0.75 + UP * y_offset)

        self.play(DrawBorderThenFill(comb))
        self.play(LaggedStart(*[Create(r) for r in rays_in], lag_ratio=0.1))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "A hologram acts as a complex microscopic diffraction grating."
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(COLOR_DIFFRACTION)

        # Diffraction rays (bending/diverging after passing through the comb gaps)
        rays_out = VGroup()
        angles = [-PI/8, 0, PI/8] # Divergence angles
        for i in range(5):
            start_p = rays_in[i].get_end()
            fan = VGroup(*[
                Line(start_p, start_p + np.array([np.cos(a), np.sin(a), 0]) * 0.8, 
                     color=COLOR_DIFFRACTION, stroke_width=1.5)
                for a in angles
            ])
            rays_out.add(fan)

        # Issue 30 Fix: 'Diffraction' label placed at A3 with scale_factor=0.8 to avoid overlap
        label_diff = Text("Diffraction", font_size=18, color=COLOR_DIFFRACTION)
        self.place_at_grid(label_diff, 'A3', scale_factor=0.8)

        self.play(
            Create(rays_out),
            Write(label_diff),
            comb.animate.set_color(COLOR_DIFFRACTION)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "This grating reconstructs the original wavefront when illuminated."
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(COLOR_DOTS)

        # Target wall on the far right - Using Asset
        wall = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/wall.svg", color=GREY_B)
        # Using columns 5-6 area for the wall to ensure it's distant enough
        self.place_in_area(wall, "A6", "F6", scale_factor=2.0)
        
        # Extend the diffracted rays to reach the wall
        rays_extended = VGroup()
        wall_x = wall.get_left()[0] # The x-coordinate of the wall's front surface
        for fan in rays_out:
            for ray in fan:
                start = ray.get_start()
                end = ray.get_end()
                direction = (end - start)
                if abs(direction[0]) < 1e-6: continue
                # Project the ray until it hits the wall_x plane
                t = (wall_x - start[0]) / direction[0]
                ext_end = start + t * direction
                rays_extended.add(Line(end, ext_end, color=COLOR_DIFFRACTION, stroke_width=1, stroke_opacity=0.5))

        # Reconstructed pattern: light dots where rays hit the wall
        dots = VGroup()
        for r_ext in rays_extended:
            dots.add(Dot(r_ext.get_end(), radius=0.05, color=COLOR_DOTS))

        self.play(FadeIn(wall))
        self.play(Create(rays_extended), run_time=1.5)
        self.play(LaggedStart(*[FadeIn(d, scale=0.5) for d in dots], lag_ratio=0.05))
        
        self.wait(3)
