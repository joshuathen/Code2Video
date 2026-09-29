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

class Section2Scene(TeachingScene):
    def construct(self):
        self.setup_layout("The Kolmogorov Scale: The Smallest Fingerprint", [
            "Kolmogorov hypothesized local small-scale isotropy.",
            "The Kolmogorov scale is where viscosity dominates.",
            "It is defined by ν and energy dissipation."
        ])
        
        # Load Assets
        fluid_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/fluid.svg")
        microscope_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/microscope.svg")
        
        vortex = Star(n=5, color="#00FF00", fill_opacity=0.5)
        small_vortices = VGroup(*[Circle(radius=0.1, color="#00FFFF", fill_opacity=0.5) for _ in range(5)]).arrange(RIGHT)
        dots = VGroup(*[Dot(color="#00FF00", radius=0.03) for _ in range(15)]).arrange_in_grid(3, 5)

        # === Animation for Lecture Line 1 ===
        self.place_at_grid(vortex, 'D3', scale_factor=0.8)
        self.place_at_grid(fluid_icon, 'C3', scale_factor=0.5)
        fluid_icon.next_to(vortex, UP, buff=0.1)
        
        self.play(FadeIn(vortex), FadeIn(fluid_icon))
        self.lecture[0].set_color("#00FF00")

        # === Animation for Lecture Line 2 ===
        self.place_at_grid(small_vortices, 'E4', scale_factor=0.9)
        self.play(Transform(vortex, small_vortices))
        self.play(FadeOut(fluid_icon))
        self.lecture[1].set_color("#00FFFF")

        # === Animation for Lecture Line 3 ===
        self.place_at_grid(dots, 'E5', scale_factor=1.0)
        self.place_at_grid(microscope_icon, 'E3', scale_factor=0.5)
        microscope_icon.next_to(dots, UP, buff=0.1)
        
        self.play(FadeIn(dots), FadeIn(microscope_icon))
        self.lecture[2].set_color("#00FF00")
        self.wait(2)
