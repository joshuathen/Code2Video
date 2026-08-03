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
        # Data from storyboard
        lecture_lines = [
            "- At the smallest scales, viscosity becomes dominant again.",
            "- Kinetic energy finally converts into vibrating heat.",
            "- The Kolmogorov length eta marks this dissipation limit."
        ]
        self.setup_layout("The Dissipation Scale (Kolmogorov Microscales)", lecture_lines)
        
        # Color palette
        VISCOSITY_COLOR = "#00FF00"
        HEAT_COLOR = "#FF0000"
        ETA_COLOR = "#D3D3D3"
        EDDY_COLOR = BLUE_B

        # Assets
        NODE_ASSET = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/node.svg"
        SURFACE_ASSET = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/surface.svg"

        # === Animation for Lecture Line 1 ===
        # Highlight: viscosity dominant
        self.play(self.lecture[0].animate.set_color(VISCOSITY_COLOR))
        
        # Represent eddies with [Asset: .../node.svg]
        try:
            eddy_template = SVGMobject(NODE_ASSET).set_color(EDDY_COLOR)
        except:
            eddy_template = Circle(radius=0.15, color=EDDY_COLOR, fill_opacity=0.5)
            
        eddies = VGroup()
        rotation_speed = ValueTracker(2.5)
        
        # Populate area C2-E5 with rotating eddies
        tl = self.grid["C2"]
        br = self.grid["E5"]
        for i in range(8):
            eddy = eddy_template.copy()
            eddy.scale(0.4)
            # Randomized position within C2-E5 bounds
            pos = np.array([
                np.random.uniform(tl[0], br[0]),
                np.random.uniform(br[1], tl[1]),
                0
            ])
            eddy.move_to(pos)
            # persistent rotation via updater
            eddy.add_updater(lambda m, dt: m.rotate(dt * rotation_speed.get_value()))
            eddies.add(eddy)
            
        self.play(FadeIn(eddies))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight: energy to heat
        self.play(self.lecture[1].animate.set_color(HEAT_COLOR))
        
        # Slow down and turn red
        self.play(
            rotation_speed.animate.set_value(0.3),
            eddies.animate.set_color(HEAT_COLOR),
            run_time=2
        )
        
        # Display text "Kinetic Energy -> Heat" in B3-B5 (Issue 35)
        energy_to_heat = Text("Kinetic Energy -> Heat", color=HEAT_COLOR, font_size=24)
        self.place_in_area(energy_to_heat, "B3", "B5", scale_factor=0.8)
        self.play(FadeIn(energy_to_heat))

        # Show metal surface [Asset: .../surface.svg]
        try:
            metal_surface = SVGMobject(SURFACE_ASSET).set_color(GRAY_B)
        except:
            metal_surface = Rectangle(width=3.5, height=2.2, color=GRAY_B, fill_opacity=0.1)
            
        self.place_in_area(metal_surface, "C2", "E5", scale_factor=1.2)
        
        # Vibrating heat dots
        num_dots = 20
        heat_dots = VGroup()
        surface_center = metal_surface.get_center()
        
        for _ in range(num_dots):
            dot = Dot(radius=0.04, color=HEAT_COLOR)
            d_pos = surface_center + np.array([
                np.random.uniform(-1.2, 1.2),
                np.random.uniform(-0.8, 0.8),
                0
            ])
            dot.move_to(d_pos)
            dot.initial_pos = d_pos
            heat_dots.add(dot)
            
        def vibrate_dots(m, dt):
            for d in m:
                d.move_to(d.initial_pos + np.array([
                    np.random.uniform(-0.04, 0.04),
                    np.random.uniform(-0.04, 0.04),
                    0
                ]))
        
        heat_dots.add_updater(vibrate_dots)
        
        self.play(
            FadeOut(eddies),
            FadeIn(metal_surface),
            FadeIn(heat_dots),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Highlight: Kolmogorov scale
        self.play(self.lecture[2].animate.set_color(ETA_COLOR))
        
        # Display eta label in F3-F5 (Issue 36)
        eta_label = Text("eta (Kolmogorov Scale)", color=ETA_COLOR, font_size=24)
        self.place_in_area(eta_label, "F3", "F5", scale_factor=0.8)
        
        self.play(FadeIn(eta_label))
        self.wait(2)
        
        # Cleanup
        for eddy in eddies:
            eddy.clear_updaters()
        heat_dots.clear_updaters()
        self.wait(1)
