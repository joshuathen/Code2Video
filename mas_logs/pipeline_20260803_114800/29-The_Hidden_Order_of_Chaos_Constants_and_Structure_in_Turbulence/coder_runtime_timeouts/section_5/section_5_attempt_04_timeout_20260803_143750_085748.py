from manim import *
import numpy as np

class TeachingScene(Scene):
    def setup_layout(self, title_text, lecture_lines):
        # BASE
        self.camera.background_color = "#000000"
        self.title = Text(title_text, font_size=28, color=WHITE).to_edge(UP)
        self.add(self.title)

        # Left-side lecture content
        lecture_texts = [Text(line, font_size=22, color=WHITE) for line in lecture_lines]
        self.lecture = VGroup(*lecture_texts).arrange(DOWN, aligned_edge=LEFT, buff=0.4).scale(0.8)
        self.lecture.to_edge(LEFT, buff=0.5)
        self.add(self.lecture)

        # Define fine-grained animation grid (6x6 grid on right side)
        self.grid = {}
        rows = ["A", "B", "C", "D", "E", "F"]
        cols = ["1", "2", "3", "4", "5", "6"]

        for i, row in enumerate(rows):
            for j, col in enumerate(cols):
                # Mapping grid to the right half of the screen
                x = 1.0 + j * 0.8
                y = 2.0 - i * 0.8
                self.grid[f"{row}{col}"] = np.array([x, y, 0])

    def place_at_grid(self, mobject, grid_pos, scale_factor=1.0):
        mobject.scale(scale_factor)
        mobject.move_to(self.grid[grid_pos])
        return mobject

    def place_in_area(self, mobject, top_left, bottom_right, scale_factor=1.0):
        tl_pos = self.grid[top_left]
        br_pos = self.grid[bottom_right]
        center = (tl_pos + br_pos) / 2
        mobject.scale(scale_factor)
        mobject.move_to(center)
        return mobject

class Section5Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "At the smallest scales, viscosity becomes dominant again.",
            "Kinetic energy finally converts into vibrating heat.",
            "The Kolmogorov length eta marks this dissipation limit."
        ]
        self.setup_layout("The Dissipation Scale (Kolmogorov Microscales)", lecture_lines)
        
        # --- Animation 1: Viscosity & Eddies ---
        # Highlight "viscosity"
        self.play(self.lecture[0].animate.set_color("#00FF00"), run_time=1)
        
        node_asset_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/node.svg"
        speed_tracker = ValueTracker(4.0)
        
        eddies = VGroup()
        for _ in range(12):
            try:
                eddy = SVGMobject(node_asset_path).set_color(BLUE_B)
            except:
                eddy = Circle(radius=0.2, color=BLUE_B)
            
            eddy.scale(np.random.uniform(0.1, 0.2))
            pos = np.array([
                np.random.uniform(1.5, 4.5),
                np.random.uniform(-1.5, 1.5),
                0
            ])
            eddy.move_to(pos)
            # Add rotation updater
            eddy.add_updater(lambda m, dt: m.rotate(dt * speed_tracker.get_value()))
            eddies.add(eddy)
        
        eddies.save_state()
        eddies.scale(0.1)
        self.play(Restore(eddies), run_time=2)
        self.wait(1)

        # --- Animation 2: Heat Conversion ---
        # Highlight second line
        self.play(self.lecture[1].animate.set_color("#FF0000"), run_time=1)
        
        # Slow down and turn red
        self.play(
            eddies.animate.set_color("#FF0000"),
            speed_tracker.animate.set_value(0.5),
            run_time=2
        )
        
        # Text "Kinetic Energy -> Heat"
        energy_to_heat = Text("Kinetic Energy -> Heat", color="#FF0000", font_size=24)
        energy_to_heat_glow = energy_to_heat.copy().set_stroke(color="#FF0000", width=4, opacity=0.4)
        energy_to_heat_group = VGroup(energy_to_heat_glow, energy_to_heat)
        self.place_in_area(energy_to_heat_group, "A3", "A5", scale_factor=0.9)
        
        self.play(Write(energy_to_heat_group), run_time=1)
        
        # Metal Surface & Vibrating Heat Dots
        surface_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/surface.svg"
        try:
            metal_surface = SVGMobject(surface_path).set_color(GRAY)
        except:
            metal_surface = Rectangle(width=3, height=2, color=GRAY, fill_opacity=0.2)
            
        self.place_in_area(metal_surface, "C3", "E5", scale_factor=1.5)
        
        heat_dots = VGroup(*[Dot(radius=0.03, color="#FF0000") for _ in range(35)])
        surface_center = metal_surface.get_center()
        
        for dot in heat_dots:
            dot.move_to(surface_center + np.array([
                np.random.uniform(-1.0, 1.0),
                np.random.uniform(-0.8, 0.8),
                0
            ]))
            dot.original_pos = dot.get_center().copy()
            dot.add_updater(lambda m: m.move_to(
                m.original_pos + np.array([
                    np.random.uniform(-0.06, 0.06),
                    np.random.uniform(-0.06, 0.06),
                    0
                ])
            ))
            
        self.play(
            FadeOut(eddies),
            FadeIn(metal_surface),
            FadeIn(heat_dots),
            run_time=1.5
        )
        self.wait(1)

        # --- Animation 3: Kolmogorov Scale ---
        self.play(self.lecture[2].animate.set_color("#D3D3D3"), run_time=1)
        
        eta_label = Text("eta (Kolmogorov Scale)", color="#D3D3D3", font_size=24)
        self.place_in_area(eta_label, "F3", "F5", scale_factor=0.8)
        
        self.play(FadeIn(eta_label), run_time=1)
        
        # Emphasis pulse
        self.play(eta_label.animate.scale(1.2).set_color(WHITE), run_time=0.5)
        self.play(eta_label.animate.scale(1/1.2).set_color("#D3D3D3"), run_time=0.5)
        
        self.wait(3)