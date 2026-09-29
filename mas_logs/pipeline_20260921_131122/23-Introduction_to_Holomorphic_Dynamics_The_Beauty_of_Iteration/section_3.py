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

class Section3Scene(TeachingScene):
    def construct(self):
        lecture_lines = ["The Julia set is a chaotic boundary.", "It separates points trapped from those escaping.", "Points inside are stable; outside ones diverge."]
        self.setup_layout("The Julia Set: The Boundary of Chaos", lecture_lines)
        
        # Assets
        microscope = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/microscope.svg", color=WHITE)
        globe = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/globe.svg", color=WHITE)

        # --- Animation Content ---
        formula = MathTex("z_{n+1} = z_n^2 + c", color=WHITE)
        self.place_in_area(formula, 'A2', 'A5', scale_factor=0.9)
        self.place_at_grid(microscope, 'A6', scale_factor=0.5)
        
        particle_cluster = VGroup(*[Dot(radius=0.08) for _ in range(20)])
        self.place_in_area(particle_cluster, 'D3', 'E5', scale_factor=0.7)
        
        # === Animation for Lecture Line 1 ===
        self.play(Write(formula), FadeIn(microscope))
        self.lecture[0].set_color("#FFFFFF")
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(FadeIn(particle_cluster))
        self.lecture[1].set_color("#00FF00")
        self.wait(1)
        
        # Highlight escaping points
        escaping = VGroup(*particle_cluster[10:])
        self.play(escaping.animate.set_color("#FF0000"))
        self.lecture[2].set_color("#FF0000")
        self.wait(1)
        
        # === Animation for Lecture Line 3 ===
        # Show boundary/fractal via globe
        self.place_at_grid(globe, 'F6', scale_factor=0.5)
        self.play(FadeIn(globe))
        self.wait(2)
