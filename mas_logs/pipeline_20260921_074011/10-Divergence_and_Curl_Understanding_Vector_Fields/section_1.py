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
        self.setup_layout("Introduction: The Concept of a Vector Field", [
            "Vector fields assign vectors to space points.",
            "Think of fluid flow velocity vectors.",
            "Each point has magnitude and direction."
        ])
        
        # Static grid of arrows
        grid = VGroup()
        for r in ["A", "B", "C", "D", "E", "F"]:
            for c in ["1", "2", "3", "4", "5", "6"]:
                arrow = Arrow(start=ORIGIN, end=RIGHT*0.3, color="#AAAAAA", buff=0)
                arrow.move_to(self.grid[f"{r}{c}"])
                grid.add(arrow)
        
        # Assets (loaded as SVGs)
        # Using placeholder SVG loading logic for Asset references
        try:
            particles_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/particles.svg")
        except:
            particles_icon = Dot(color="#00FFFF")
        try:
            fluid_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/fluid.svg")
        except:
            fluid_icon = Circle(color="#0000FF")

        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(grid))
        self.lecture[0].set_color("#FFCC00")
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color("#FFCC00")
        
        # Highlight a single arrow
        highlight_arrow = grid[18].copy().set_color("#FFCC00").scale(1.5)
        self.play(FadeIn(highlight_arrow))
        self.wait(1)
        self.play(FadeOut(highlight_arrow))
        
        # Show particles as per asset requirements
        # Fix: Vector Field Particles placement per issue 21
        particles_group = VGroup(*[particles_icon.copy() for _ in range(10)])
        self.place_in_area(particles_group, 'C4', 'F6', scale_factor=0.6)
        
        self.add(particles_group)
        self.wait(2)
        
        # Fade out arrows, keep particles flowing using fluid.svg
        self.play(FadeOut(grid))
        new_particles = VGroup(*[fluid_icon.copy() for _ in range(10)])
        self.place_in_area(new_particles, 'C4', 'F6', scale_factor=0.6)
        self.play(ReplacementTransform(particles_group, new_particles))
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color("#FFCC00")
        self.wait(2)
