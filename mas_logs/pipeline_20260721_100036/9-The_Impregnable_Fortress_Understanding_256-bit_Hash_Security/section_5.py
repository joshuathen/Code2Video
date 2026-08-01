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
        self.setup_layout("Energy vs. Probability", [
            "Computation requires energy and follows physical laws.",
            "A Dyson Sphere captures every spark from our Sun.",
            "Even this infinite battery cannot power the search.",
            "Computers would melt before finding the correct key.",
            "Math provides security that physics cannot break."
        ])
        
        # Colors
        COLOR_SUN = "#FFFF00"
        COLOR_SPHERE = "#A9A9A9"
        COLOR_ENERGY = "#FFA500"
        COLOR_PROCESSOR = "#FFFFFF"
        COLOR_MELT = "#FF4500"
        COLOR_WARNING = "#FF0000"

        # Assets
        SUN_ASSET = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/sun.svg"

        # === Animation for Lecture Line 1 ===
        # Computation requires energy and follows physical laws.
        self.play(self.lecture[0].animate.set_color(COLOR_SUN))
        
        sun_svg = SVGMobject(SUN_ASSET, color=COLOR_SUN, fill_opacity=1)
        sun_glow = Dot(radius=0.7, color=COLOR_SUN, fill_opacity=0.3)
        sun_group = VGroup(sun_glow, sun_svg)
        self.place_in_area(sun_group, "B3", "D5", scale_factor=0.9)
        
        self.play(FadeIn(sun_group))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # A Dyson Sphere captures every spark from our Sun.
        self.play(self.lecture[0].animate.set_color(WHITE), self.lecture[1].animate.set_color(COLOR_SPHERE))
        
        sphere = RegularPolygon(n=6, color=COLOR_SPHERE, fill_opacity=0).scale(1.2)
        sphere_lines = VGroup(*[
            Line(sphere.get_vertices()[i], sphere.get_vertices()[(i+1)%6], color=COLOR_SPHERE)
            for i in range(6)
        ])
        sphere_cross = VGroup(*[
            Line(sphere.get_vertices()[i], sphere.get_vertices()[(i+3)%6], color=COLOR_SPHERE, stroke_opacity=0.5)
            for i in range(3)
        ])
        dyson_sphere = VGroup(sphere, sphere_lines, sphere_cross)
        dyson_sphere.move_to(sun_group.get_center())
        
        self.play(Create(dyson_sphere))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Even this infinite battery cannot power the search.
        self.play(self.lecture[1].animate.set_color(WHITE), self.lecture[2].animate.set_color(COLOR_ENERGY))
        
        proc_rect = Rectangle(width=1, height=1, color=COLOR_PROCESSOR, fill_opacity=0.1)
        pins_v = VGroup(*[
            Line(start=proc_rect.get_top() + RIGHT * x, end=proc_rect.get_top() + RIGHT * x + UP * 0.2, color=COLOR_PROCESSOR)
            for x in np.linspace(-0.4, 0.4, 5)
        ], *[
            Line(start=proc_rect.get_bottom() + RIGHT * x, end=proc_rect.get_bottom() + RIGHT * x + DOWN * 0.2, color=COLOR_PROCESSOR)
            for x in np.linspace(-0.4, 0.4, 5)
        ])
        pins_h = VGroup(*[
            Line(start=proc_rect.get_left() + UP * y, end=proc_rect.get_left() + UP * y + LEFT * 0.2, color=COLOR_PROCESSOR)
            for y in np.linspace(-0.4, 0.4, 5)
        ], *[
            Line(start=proc_rect.get_right() + UP * y, end=proc_rect.get_right() + UP * y + RIGHT * 0.2, color=COLOR_PROCESSOR)
            for y in np.linspace(-0.4, 0.4, 5)
        ])
        processor = VGroup(proc_rect, pins_v, pins_h)
        self.place_at_grid(processor, "F6", scale_factor=0.8)
        
        self.play(FadeIn(processor))
        
        energy_line = Line(dyson_sphere.get_right(), processor.get_left(), color=COLOR_ENERGY, stroke_width=4)
        
        energy_pulse = energy_line.copy().set_stroke(width=6)
        def pulse_updater(m, dt):
            m.set_stroke(opacity=0.5 + 0.5 * np.sin(self.renderer.time * 7))
            
        energy_pulse.add_updater(pulse_updater)
        
        self.play(Create(energy_line))
        self.add(energy_pulse)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Computers would melt before finding the correct key.
        self.play(self.lecture[2].animate.set_color(WHITE), self.lecture[3].animate.set_color(COLOR_MELT))
        
        self.play(
            proc_rect.animate.set_fill(COLOR_MELT, opacity=0.8).set_color(COLOR_MELT),
            pins_v.animate.set_color(COLOR_MELT),
            pins_h.animate.set_color(COLOR_MELT),
            energy_line.animate.set_color(COLOR_MELT)
        )
        
        # Smoke effect
        smoke_particles = VGroup(*[
            Dot(radius=0.05, color=GRAY, fill_opacity=0.6).move_to(
                processor.get_center() + np.array([np.random.uniform(-0.3, 0.3), np.random.uniform(-0.1, 0.1), 0])
            )
            for _ in range(12)
        ])
        
        def smoke_logic(m, dt):
            for part in m:
                part.shift(UP * dt * 0.6 + RIGHT * np.sin(self.renderer.time * 2 + part.get_center()[1]) * dt * 0.2)
                # Fix: Use get_fill_opacity and set_fill_opacity instead of set_opacity/get_opacity
                new_opacity = max(0, part.get_fill_opacity() - dt * 0.3)
                part.set_fill_opacity(new_opacity)
                if new_opacity <= 0:
                    part.move_to(processor.get_center() + np.array([np.random.uniform(-0.3, 0.3), np.random.uniform(-0.1, 0.1), 0]))
                    part.set_fill_opacity(0.6)

        smoke_particles.add_updater(smoke_logic)
        self.add(smoke_particles)
        self.wait(2)

        # === Animation for Lecture Line 5 ===
        # Math provides security that physics cannot break.
        self.play(self.lecture[3].animate.set_color(WHITE), self.lecture[4].animate.set_color(COLOR_WARNING))
        
        warning_text = Text("SYSTEM FAILURE", font_size=36, color=COLOR_WARNING, weight=BOLD)
        self.place_at_grid(warning_text, "E5", scale_factor=1.0)
        
        for _ in range(3):
            self.play(FadeIn(warning_text), run_time=0.3)
            self.play(FadeOut(warning_text), run_time=0.3)
            
        self.play(FadeIn(warning_text))
        self.wait(2)

        self.play(
            FadeOut(sun_group),
            FadeOut(dyson_sphere),
            FadeOut(energy_line),
            FadeOut(energy_pulse),
            FadeOut(processor),
            FadeOut(smoke_particles),
            FadeOut(warning_text),
            self.lecture[4].animate.set_color(WHITE)
        )
        self.wait(1)
