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

class Section1Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Turbulence is an energy cascade of chaotic fluid motion.",
            "Inertial forces dominate at high velocities.",
            "Viscous forces eventually dampen the flow motion."
        ]
        self.setup_layout("Introduction: The Chaos in the Fluid", lecture_lines)
        
        # Assets
        fluid_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/fluid.svg")
        particles_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/particles.svg")

        # Setup visuals
        turbulent_field = VGroup(*[
            Arrow(start=ORIGIN, end=RIGHT*0.3, color="#3498DB").shift(self.grid[f"{r}{c}"])
            for r in "CDE" for c in "456"
        ])
        
        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(turbulent_field), self.lecture[0].animate.set_color("#3498DB"))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Using asset: [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/fluid.svg]
        self.place_at_grid(fluid_icon, "B3", scale_factor=0.5)
        self.play(self.lecture[1].animate.set_color("#E74C3C"), FadeIn(fluid_icon))
        # Animate pushing out
        self.play(fluid_icon.animate.scale(1.5).set_color("#E74C3C"))
        
        # === Animation for Lecture Line 3 ===
        # Using asset: [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/particles.svg]
        self.place_at_grid(particles_icon, "D4", scale_factor=0.5)
        dissipation_zone = Rectangle(color="#2ECC71", fill_opacity=0.3, width=2, height=2)
        self.place_in_area(dissipation_zone, "D4", "F6", scale_factor=0.6)
        
        self.play(
            self.lecture[2].animate.set_color("#2ECC71"), 
            FadeIn(particles_icon),
            FadeIn(dissipation_zone)
        )
        self.play(FadeOut(particles_icon))
        self.wait(2)
