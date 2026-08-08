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

class Section4Scene(TeachingScene):
    def construct(self):
        self.setup_layout(
            "Reconstruction: Diffraction as the Decoder",
            [
                "The reference beam illuminates the recorded interference pattern.",
                "The pattern acts as a complex diffraction grating.",
                "Bragg's Law governs how light bends through it.",
                "Diffracted waves reconstruct the original 3D wavefront.",
                "A holographic ghost image appears in three dimensions."
            ]
        )

        # Colors
        COLOR_BEAM = "#00FF00"
        COLOR_PLATE = "#00FFFF"
        COLOR_BRAGG = "#FFFF00"
        COLOR_WAVEFRONT = "#00FFFF"
        COLOR_STATUE = "#FFFFFF"

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(COLOR_BEAM))
        
        # Holographic Plate at Column 4
        plate = Rectangle(height=4, width=0.1, color=COLOR_PLATE, fill_opacity=0.3)
        self.place_in_area(plate, 'B4', 'E4')
        
        # Reference Beam (Incoming parallel lines) from Column 2 to 4
        beam_lines = VGroup(*[
            Line(self.grid['B2'] + UP*i*0.4, self.grid['B4'] + UP*i*0.4, color=COLOR_BEAM, stroke_width=2)
            for i in range(-5, 6)
        ])
        
        self.play(Create(plate))
        self.play(Create(beam_lines), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(COLOR_PLATE))
        
        # Zoom into plate to show microscopic slits
        zoom_plate_bg = Rectangle(height=4.2, width=0.3, color=COLOR_PLATE, fill_opacity=0.1).move_to(plate)
        slits = VGroup(*[
            Rectangle(height=0.1, width=0.2, color=WHITE, fill_opacity=0.8).move_to(plate.get_center() + UP*i*0.4)
            for i in range(-5, 6)
        ])
        
        self.play(
            FadeOut(plate),
            FadeOut(beam_lines),
            FadeIn(zoom_plate_bg),
            FadeIn(slits)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(COLOR_BRAGG))
        
        # Show rays bending according to Bragg's Law
        bending_rays = VGroup()
        for slit in slits:
            start_pos = slit.get_center()
            # Bending at an angle (d sin theta = n lambda)
            # Offset to Column 6
            end_pos = start_pos + np.array([2.2, 0.8, 0]) 
            ray = Line(start_pos, end_pos, color=COLOR_BRAGG, stroke_width=2).add_tip(tip_length=0.1)
            bending_rays.add(ray)
            
        bragg_formula = MathTex(r"d \sin \theta = n \lambda", color=COLOR_BRAGG)
        # Issue 40: Move bragg_formula to 'A6' and scale factor 0.7
        self.place_at_grid(bragg_formula, 'A6', scale_factor=0.7)
        
        self.play(Write(bragg_formula))
        self.play(Create(bending_rays), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(COLOR_WAVEFRONT))
        
        # Reconstruct wavefront (arcs appearing on the exit side)
        reconstruction_center = self.grid['D6'] + LEFT*0.5
        wavefronts = VGroup(*[
            Arc(radius=0.5 + i*0.4, start_angle=-PI/3, angle=2*PI/3, color=COLOR_WAVEFRONT, stroke_width=2)
            .move_to(reconstruction_center)
            for i in range(4)
        ])
        
        self.play(
            bending_rays.animate.set_stroke(opacity=0.4),
            Create(wavefronts),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(COLOR_STATUE))
        
        # Issue 30: Use SVGMobject for the statue
        # Issue 41: Scale factor 0.6 at D6
        statue_mobject = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/statue.svg", color=COLOR_STATUE, fill_opacity=0.15)
        self.place_at_grid(statue_mobject, 'D6', scale_factor=0.6)
        
        # Ghostly shimmer effect using a simple updater
        # This is safe as it's not redrawing the mobject, just changing opacity
        statue_mobject.add_updater(lambda m, dt: m.set_fill(opacity=0.15 + 0.1 * np.sin(self.time * 3)))
        
        self.play(
            FadeIn(statue_mobject),
            bragg_formula.animate.set_fill(opacity=0.5),
            wavefronts.animate.set_stroke(opacity=0.5)
        )
        self.wait(3)
